#!/usr/bin/env python3
"""Build descriptive, protocol-separated downstream evidence for the paper vision."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
ANALYSIS = REPO / 'scripts/experiments/analysis'
sys.path.insert(0, str(REPO / 'scripts/experiments/setup/final_core'))
from core_plan import ORDERS

TARGETS = {'covid_political', 'election2020', 'facebook_page_reference', 'twibot20', 'ukr_rus_suspended'}
ALIASES = {'covid19_twitter': 'covid', 'ukr_rus_twitter': 'ukr_rus', 'cp_hk_twitter': 'cp_hk'}
INPUTS = []


def read(relative):
    path = ANALYSIS / relative
    INPUTS.append({'path': str(path.relative_to(REPO)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return pd.read_csv(path, sep='\t' if path.suffix == '.tsv' else ',')


def sources(value):
    items = ast.literal_eval(value) if value.startswith('[') else value.split(',')
    return tuple(sorted(ALIASES.get(x.strip(), x.strip()) for x in items))


def normalize(d, protocol, model, objective, expected):
    d = d.rename(columns={'dataset': 'target', 'seed': 'training_seed', 'checkpoint_update': 'checkpoint_step',
                          'roc_auc_mean': 'roc_auc', 'accuracy_mean': 'accuracy'}).copy()
    assert len(d) == expected, (protocol, len(d), expected)
    d = d[d.target.isin(TARGETS)].copy()
    d['source_set'] = d.sources.map(sources)
    d['sources'] = d.source_set.map(','.join)
    d['source_count'] = d.source_set.map(len)
    d['target_seen'] = [t in s for t, s in zip(d.target, d.source_set)]
    d['protocol'] = protocol
    d['model'] = model
    d['objective'] = objective
    if 'stream' not in d:
        d['stream'] = 'original'
    if 'schedule' not in d:
        d['schedule'] = 'native_mixture'
    assert not d.duplicated(['model_id', 'training_seed', 'target', 'stream']).any(), protocol
    assert np.isfinite(d[['roc_auc', 'accuracy']]).all().all()
    assert d.checkpoint_step.eq(500 if model == 'SAMGPT' else 2500).all()
    assert d[['roc_auc', 'accuracy']].ge(0).all().all() and d[['roc_auc', 'accuracy']].le(1).all().all()
    assert d.groupby(['stream', 'target']).episode_fingerprint.nunique().eq(1).all(), protocol
    assert d.groupby(['model_id', 'training_seed', 'stream']).target.nunique().eq(4 if model == 'SAMGPT' else 5).all(), protocol
    return d


def main():
    base = 'transfer/matrices/cross_model/final_core/data/'
    native = 'synthesis/cross_experiment/native_model_result_matrix/data/'
    p = normalize(read(base + 'classification_ladder/classification_long.tsv'), 'prodigy_ladder', 'PRODIGY', 'neighbor_matching', 375)
    s = normalize(read(base + 'samgpt_downstream_cls/cells.csv'), 'samgpt_ladder', 'SAMGPT', 'GraphCL', 837)
    v = read(native + 'vision_native_mixture_cells.csv')
    v = v[v.checkpoint_step.eq(2500)].drop(columns=['seed']).copy()
    # Shared all-nine checkpoints appear under several logical order aliases.
    duplicate_key = ['model_id', 'training_seed', 'dataset']
    assert v.groupby(duplicate_key).roc_auc.nunique().eq(1).all()
    assert v.groupby(duplicate_key).accuracy.nunique().eq(1).all()
    v = normalize(v.drop_duplicates(duplicate_key), 'vision_ladder', 'VISION', 'feature_similarity', 65)
    lattice = read('graphs/transfer_prediction/target_performance_mechanisms/data/raw/classification_long.tsv').drop(columns=['seed'])
    l = normalize(lattice, 'prodigy_lattice', 'PRODIGY', 'neighbor_matching', 270)
    q = read('graphs/transfer_prediction/trace_schedule_scaling/data/cells.csv')
    q['checkpoint_step'] = 2500
    q = normalize(q, 'prodigy_schedule', 'PRODIGY', 'neighbor_matching', 270)
    cells = pd.concat([p, s, v, l, q], ignore_index=True)
    cells['budget_regime'] = 'fixed_updates'
    cells['source_path'] = cells.protocol.map(dict(zip(
        ['prodigy_ladder', 'samgpt_ladder', 'vision_ladder', 'prodigy_lattice', 'prodigy_schedule'],
        [x['path'] for x in INPUTS])))
    catalog = json.loads((REPO / 'docs/graph_catalog.json').read_text())
    canonical = {g['dataset_key']: g['canonical_name'] for g in catalog['graphs']}
    cells['target_canonical_name'] = cells.target.map(canonical)
    assert cells.target_canonical_name.notna().all()
    columns = ['protocol', 'model', 'objective', 'model_id', 'sources', 'source_count', 'target',
               'target_canonical_name', 'target_seen', 'training_seed', 'checkpoint_step', 'budget_regime',
               'stream', 'schedule', 'episode_fingerprint', 'roc_auc', 'accuracy', 'source_path']
    out = HERE / 'data'
    out.mkdir(parents=True, exist_ok=True)
    cells[columns].to_csv(out / 'downstream_cells.csv', index=False)

    comparisons = []

    def contrast(row, reference, kind, order='', previous_count=None, added=''):
        assert row.episode_fingerprint == reference.episode_fingerprint
        assert row.training_seed == reference.training_seed and row.stream == reference.stream
        role = 'incumbent' if row.target_seen and reference.target_seen else ('newcomer' if row.target_seen else 'unseen')
        for metric in ('roc_auc', 'accuracy'):
            comparisons.append({
                'protocol': row.protocol, 'model': row.model, 'objective': row.objective,
                'comparison': kind, 'order': order, 'source_count': row.source_count,
                'previous_source_count': previous_count, 'added_sources': added,
                'model_id': re.sub(r'_s\d+$', '', row.model_id) if row.protocol == 'prodigy_schedule' else row.model_id,
                'physical_model_id': row.model_id, 'reference_model_id': reference.model_id,
                'sources': row.sources, 'target': row.target, 'target_seen': row.target_seen,
                'role': role, 'training_seed': row.training_seed, 'checkpoint_step': row.checkpoint_step,
                'stream': row.stream, 'metric': metric, 'value': row[metric],
                'reference_value': reference[metric], 'gain': row[metric] - reference[metric],
            })

    for frame in (p, s, v):
        for order, sequence in ORDERS.items():
            sizes = [1, 3, 5, 7, 9] if frame is v else list(range(1, 10))
            rungs = {k: frame[frame.source_set.map(lambda x: x == tuple(sorted(sequence[:k])))] for k in sizes}
            for k, rung in rungs.items():
                assert len(rung) == frame.target.nunique() * frame.training_seed.nunique(), (frame.protocol.iloc[0], order, k)
            for i, k in enumerate(sizes[1:], 1):
                for _, row in rungs[k].iterrows():
                    for reference_k, kind in ((sizes[i - 1], 'adjacent_rung'), (1, 'versus_first_source')):
                        ref = rungs[reference_k]
                        ref = ref[(ref.training_seed == row.training_seed) & (ref.target == row.target)].iloc[0]
                        contrast(row, ref, kind, order, reference_k, ','.join(sequence[reference_k:k]))

    for _, row in l[l.source_count.gt(1)].iterrows():
        singles = l[(l.source_count == 1) & (l.target == row.target) & (l.training_seed == row.training_seed)]
        singles = singles[singles.source_set.map(lambda x: x[0] in row.source_set)]
        assert len(singles) == row.source_count
        # Best constituent is selected separately for each metric: descriptive oracle.
        for metric in ('roc_auc', 'accuracy'):
            ref = singles.loc[singles[metric].idxmax()]
            contrast(row, ref, 'versus_best_constituent')
            comparisons.pop(-1 if metric == 'roc_auc' else -2)

    for _, row in q[q.schedule.ne('interleaved')].iterrows():
        ref = q[(q.schedule == 'interleaved') & (q.source_count == row.source_count) &
                (q.target == row.target) & (q.training_seed == row.training_seed) & (q.stream == row.stream)]
        assert len(ref) == 1
        contrast(row, ref.iloc[0], row.schedule + '_versus_interleaved')

    deltas = pd.DataFrame(comparisons)
    deltas.to_csv(out / 'paired_gains.csv', index=False)
    # Seed-average each paired target comparison FIRST. Seeds are not extra targets.
    group = ['protocol', 'model', 'comparison', 'order', 'source_count', 'previous_source_count',
             'model_id', 'reference_model_id', 'target', 'target_seen', 'role', 'checkpoint_step', 'stream', 'metric']
    # For best-constituent comparisons the oracle identity can vary across seeds.
    group.remove('reference_model_id')
    means = deltas.groupby(group, dropna=False).agg(gain=('gain', 'mean'), training_seeds=('training_seed', 'nunique'),
                seed_min_gain=('gain', 'min'), seed_max_gain=('gain', 'max')).reset_index()
    expected_seeds = {'prodigy_ladder': 3, 'samgpt_ladder': 3, 'vision_ladder': 1, 'prodigy_lattice': 1, 'prodigy_schedule': 3}
    assert means.training_seeds.eq(means.protocol.map(expected_seeds)).all()
    means.to_csv(out / 'target_gains.csv', index=False)
    model_summaries = []
    model_keys = ['protocol', 'model', 'comparison', 'order', 'source_count', 'model_id', 'checkpoint_step', 'stream', 'metric']
    for key, frame in means.groupby(model_keys, dropna=False):
        for scope in ('all', 'in_mix', 'unseen'):
            part = frame if scope == 'all' else frame[frame.target_seen.eq(scope == 'in_mix')]
            model_summaries.append(dict(zip(model_keys, key)) | {
                'scope': scope, 'targets': len(part), 'mean_gain': part.gain.mean(),
                'fraction_targets_improved': part.gain.gt(1e-12).mean() if len(part) else None,
                'worst_gain': part.gain.min(), 'worst_target': part.loc[part.gain.idxmin(), 'target'] if len(part) else '',
            })
    pd.DataFrame(model_summaries).to_csv(out / 'model_metrics.csv', index=False)
    records = []
    summary_keys = ['protocol', 'model', 'comparison', 'order', 'source_count', 'previous_source_count',
                    'checkpoint_step', 'stream', 'metric']
    for key, frame in means.groupby(summary_keys, dropna=False):
        for scope in ('all', 'in_mix', 'unseen', 'incumbent', 'newcomer', 'common4_all', 'common4_in_mix', 'common4_unseen'):
            part = frame if scope == 'all' else frame[frame.target_seen.eq(scope == 'in_mix')] if scope in ('in_mix', 'unseen') else frame[frame.role.eq(scope)]
            if scope.startswith('common4_'):
                part = frame[frame.target.ne('facebook_page_reference')]
                if scope != 'common4_all':
                    part = part[part.target_seen.eq(scope == 'common4_in_mix')]
            records.append(dict(zip(summary_keys, key)) | {
                'scope': scope, 'target_comparisons': len(part), 'distinct_targets': part.target.nunique(),
                'training_seeds_min': part.training_seeds.min() if len(part) else None,
                'mean_gain': part.gain.mean(), 'fraction_improved': part.gain.gt(1e-12).mean() if len(part) else None,
                'worst_gain': part.gain.min(), 'best_gain': part.gain.max(),
                'worst_target': part.loc[part.gain.idxmin(), 'target'] if len(part) else '',
            })
    summary = pd.DataFrame(records)
    # Reconcile against independently produced prior analyses, including seed averaging.
    published = read('graphs/transfer_prediction/trace_schedule_scaling/data/contrast_summary_by_target.csv')
    schedule_means = means[(means.protocol == 'prodigy_schedule') & (means.metric == 'roc_auc')].copy()
    schedule_means['schedule'] = schedule_means.comparison.str.replace('_versus_interleaved', '', regex=False)
    matched = schedule_means.merge(published, left_on=['stream', 'target', 'source_count', 'schedule'],
        right_on=['stream', 'target', 'rung', 'schedule'], validate='one_to_one')
    assert len(matched) == 60 and np.allclose(matched.gain, matched.mean_delta_roc_auc, atol=1e-12, rtol=0)
    foreign_pairs = summary[(summary.protocol == 'prodigy_lattice') & (summary.scope == 'unseen') &
                            (summary.source_count == 2) & (summary.metric == 'roc_auc')].iloc[0]
    assert foreign_pairs.target_comparisons == 140
    assert abs(foreign_pairs.mean_gain - (-0.01076)) < 0.000005
    assert abs(foreign_pairs.fraction_improved - 2 / 7) < 1e-12
    summary.to_csv(out / 'deck_metrics.csv', index=False)
    coverage = cells.groupby(['protocol', 'model', 'checkpoint_step', 'stream']).agg(
        cells=('roc_auc', 'size'), models=('model_id', 'nunique'), targets=('target', 'nunique'),
        training_seeds=('training_seed', 'nunique'), minimum_sources=('source_count', 'min'), maximum_sources=('source_count', 'max')).reset_index()
    coverage.to_csv(out / 'coverage.csv', index=False)
    perf = cells.copy()
    perf['model_id'] = [re.sub(r'_s\d+$', '', m) if p == 'prodigy_schedule' else m for m, p in zip(perf.model_id, perf.protocol)]
    perf = perf.groupby(['protocol', 'model', 'model_id', 'source_count', 'target', 'target_seen', 'stream', 'schedule'], dropna=False)[['roc_auc', 'accuracy']].mean().reset_index()
    perf.to_csv(out / 'target_performance.csv', index=False)
    performance = []
    for key, frame in perf.groupby(['protocol', 'model', 'source_count', 'stream', 'schedule']):
        for scope in ('all', 'in_mix', 'unseen', 'common4_all'):
            part = frame if scope == 'all' else frame[frame.target.ne('facebook_page_reference')] if scope == 'common4_all' else frame[frame.target_seen.eq(scope == 'in_mix')]
            performance.append(dict(zip(['protocol', 'model', 'source_count', 'stream', 'schedule'], key)) | {
                'scope': scope, 'target_comparisons': len(part), 'mean_roc_auc': part.roc_auc.mean(),
                'mean_accuracy': part.accuracy.mean(), 'minimum_roc_auc': part.roc_auc.min()})
    pd.DataFrame(performance).to_csv(out / 'performance_by_size.csv', index=False)
    receipt = {'inputs': INPUTS, 'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
               'branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip(),
               'cells': len(cells), 'paired_metric_rows': len(deltas), 'summary_rows': len(summary),
               'checks': ['expected input coverage', 'unique physical model/seed/target/stream cells', 'registered target panels (SAMGPT common four)',
                          'finite bounded outcomes', 'within-protocol fingerprints', 'paired fingerprints and seeds', 'complete ladder rungs',
                          'expected seed counts after averaging', '60 schedule AUC means reproduce original analysis', 'foreign-pair published summary reproduced'],
               'catalog_sha256': hashlib.sha256((REPO / 'docs/graph_catalog.json').read_bytes()).hexdigest()}
    (out / 'validation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    def table(frame):
        rows = ['| ' + ' | '.join(frame.columns) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
        for row in frame.itertuples(index=False, name=None):
            rows.append('| ' + ' | '.join('—' if pd.isna(x) else f'{x:.4f}' if isinstance(x, float) else str(x) for x in row) + ' |')
        return '\n'.join(rows)
    endpoint = summary[(summary.comparison == 'versus_first_source') & (summary.source_count == 9) &
                       (summary.metric == 'roc_auc') & (summary.scope == 'common4_all')]
    lattice_summary = summary[(summary.comparison == 'versus_best_constituent') & (summary.metric == 'roc_auc') & (summary.scope == 'unseen')]
    schedule_summary = summary[(summary.protocol == 'prodigy_schedule') & (summary.metric == 'roc_auc') & (summary.scope == 'all') & (summary.stream == 'fresh')]
    report = '''# Downstream evidence for the multi-graph pretraining paper

This audit assembles completed local result exports for slides 3 and 7 of
`SGFM Paper Presentation(1).pptx`. It launches no experiments and does not verify
live Tucker state. The evidence supports target-dependent benefits from adding
graphs; it does not establish a generally non-degrading pretraining framework.

## Coverage

'''+table(coverage)+'''

There are **1,352 protocol-specific physical model–seed–target–stream cells**.
The same checkpoint can occur in separate experiments; these are not 1,352
independent observations. Shared ladder endpoints are deduplicated in the cell
table and reused only in explicitly named order comparisons.

SAMGPT's native export uses different Facebook label tasks. Its five nonmatching
targets (three Facebook tasks, Cora, PubMed) are excluded here. The common panel
is COVID political, Election 2020, TwiBot-20, and Ukraine suspended. Other panels
also retain Facebook Page Reference. Even on the common panel, model-native
evaluation protocols and compute differ: compare within-model gains, not a
pooled cross-model score or an equal-compute architecture ranking.

## RQ2: all-nine versus each order's first specialist

The following AUC differences average training seeds first, then the four common
targets. Fraction improved and worst gain also use these seed-mean target gains.
Orders change which sources belong to each rung; they are not temporal training
schedules. All-nine has no unseen targets in this panel, so the unseen endpoint
metric is undefined, not zero. Consult earlier rungs in `data/deck_metrics.csv`
for genuinely unseen-target changes.

'''+table(endpoint[['model', 'order', 'mean_gain', 'fraction_improved', 'worst_gain', 'worst_target']])+'''

The endpoint direction depends on the starting specialist. The same all-nine
checkpoint is reused across A/B/C within each seed; these endpoint comparisons
are not independent replications. VISION has one training seed and only odd
rungs. Its adjacent comparisons add two sources at a time, not one.

## RQ1: mixtures versus their best constituent specialist

These are wholly foreign-source comparisons: the target graph is absent from
the mixture. Best constituent is a per-target, per-metric oracle comparator,
not a deployable source selector. Each pair has two independently trained
specialists available as references; each leave-one-out mixture has eight.

'''+table(lattice_summary[['source_count', 'target_comparisons', 'mean_gain', 'fraction_improved', 'worst_gain', 'worst_target']])+'''

Here the fraction is over mixture–target comparisons (140 for pairs), not 140
distinct targets. `data/model_metrics.csv` supplies the fraction of targets
improved for each individual mixture. There is one training seed. Mixtures and
specialists have the same total updates but different per-source exposure;
negative gains do not identify causal interference.

## RQ1: exactly matched schedule comparisons

Fresh-stream AUC gains relative to interleaving, averaged over three training
seeds and then five targets. Blocked and replay100 use the same source-private
examples as their matched interleaved reference. Original-stream results remain
separate in the data. This table reports descriptive means and worst targets;
it does not replace the original study's crossed uncertainty analysis.

'''+table(schedule_summary[['comparison', 'source_count', 'mean_gain', 'fraction_improved', 'worst_gain', 'worst_target']])+'''

## Definitions and interpretation

- `downstream_cells.csv`: outcomes, sources, target membership, training seed,
  checkpoint update budget, evaluation fingerprint, and input path.
- `target_performance.csv` and `performance_by_size.csv`: absolute AUC and
  accuracy, with seeds averaged before descriptive corpus-size summaries.
- `paired_gains.csv`: every seed-level paired metric and its exact reference.
- `target_gains.csv`: per-target seed means and observed seed ranges.
- `model_metrics.csv`: each mixture's mean gain, fraction of targets improved,
  worst gain, worst target, and in-mixture/unseen breakdown.
- `deck_metrics.csv`: order/rung or study-level aggregates, including common-four
  scopes, incumbent/newcomer roles, and adjacent-rung marginal gains.
- `coverage.csv` and `validation.json`: coverage checks and hashed inputs.

Positive gain means improvement. A win exceeds 1e-12, a numerical tie tolerance,
not a practical-effect threshold. Worst gain is the minimum seed-mean paired
gain; a negative value denotes degradation. Blank zero-target summaries are
intentional. Seed ranges are descriptive, not confidence intervals. Fixed
episode fingerprints do not create independent evaluation seeds or guarantee
identical sampled contexts across different studies. No between-protocol
contrasts are calculated.

The main results use AUC; accuracy is retained in all comparison exports.
Macro-F1 is not harmonized because it is not available consistently. Update
counts do not imply equal FLOPs, wall time, convergence, or per-source exposure.
Do not infer unseen performance from changing in-mixture/held-out panel means:
adjacent gains always pair the same target, then classify its new membership.

## Still missing from the deck

The table does not supply a downstream cross-graph sampling-ratio sweep, a
size-proportional-to-balanced exposure sweep, independent data-volume or
capacity sweeps, or a validated new pretraining framework. Earlier NM-only
controls cannot fill these downstream cells. GraphSAGE and adaptation-efficiency
results address different designs and are not spliced into these comparisons.
This is a classification evidence table, not evidence of transfer across every
downstream task; regression and valid pair-link prediction require separate
protocol-specific audits if the paper retains that broader claim.
'''
    (HERE / 'FINDINGS.md').write_text(report)
    print(coverage.to_string(index=False))
    print(json.dumps({k: receipt[k] for k in ('cells', 'paired_metric_rows', 'summary_rows')}))


if __name__ == '__main__':
    main()
