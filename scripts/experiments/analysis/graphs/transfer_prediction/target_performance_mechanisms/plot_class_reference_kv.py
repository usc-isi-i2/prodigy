"""Figure-led synthesis of completed K/V tests; no model forwards or fitting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


CONDITIONS = ['keys_only', 'values_only', 'joint_kv']
COLORS = dict(zip(CONDITIONS, ['#7964A7', '#207F83', '#C56832']))
LABELS = dict(zip(CONDITIONS, ['Keys only', 'Values only', 'Keys + values']))
SOURCES = ['covid', 'covid_political', 'cp_hk', 'election2020',
           'facebook_page_reference', 'midterm', 'twibot20', 'ukr_rus', 'ukr_rus_suspended']
SOURCE_LABELS = ['covid', 'covid-political', 'Hong Kong', 'election2020',
                 'facebook-page', 'midterm', 'twibot20', 'ukraine', 'ukraine-suspended']
TARGETS = ['covid_political', 'election2020', 'facebook_page_reference', 'twibot20', 'ukr_rus_suspended']


def read(path):
    return json.loads(path.read_text())


def mean(rows, field):
    return float(np.mean([r[field] for r in rows]))


def summarize(root):
    discovery = root / 'classrefkv_discovery_20260907'
    long = root / 'classrefkv_long_20260907'
    prediction = read(discovery / 'prediction.json')
    assert prediction == read(long / 'protocol.json')['frozen_prediction']
    assert prediction['hypothesis'] == 'values_only'
    assessments = read(long / 'prediction_assessment.json')
    assert len(assessments) == 2 and all(r['primary_prediction_passed'] for r in assessments)
    for directory, count in [(discovery, 5), (long, 10)]:
        done = read(directory / 'DONE.json')
        assert done['cells'] == count and done['all_native_endpoints_bit_exact']
        assert done['all_joint_endpoints_bit_exact'] and not done['new_training']
        assert all(r['label_attention_routes_bit_exact'] for r in read(directory / 'receipts.json'))
    cells = []
    geos = []
    receipts = {}
    for directory, label in [(discovery, 'Discovery: 2.5k'), (long, 'Prediction test: 50k')]:
        metrics = read(directory / 'metrics.json')
        geometry = read(directory / 'geometry.json')
        for stream in dict.fromkeys(r['stream'] for r in metrics):
            block = [r for r in metrics if r['stream'] == stream]
            base = next(r for r in block if r['condition'] == 'intact')
            for row in block:
                cells.append({**row, 'configuration': label,
                    'delta_within_points': 100 * (row['within_episode_auc'] - base['within_episode_auc']),
                    'delta_pooled_points': 100 * (row['pooled_margin_auc'] - base['pooled_margin_auc'])})
        for condition in ['intact', 'removed', *CONDITIONS]:
            rows = [r for r in geometry if r['condition'] == condition]
            assert len(rows) == (128 if directory == discovery else 256)
            fields = [k for k in rows[0] if k not in ['condition', 'stream', 'batch', 'episode_in_batch']]
            geos.append({'configuration': label, 'condition': condition,
                         'episodes': len(rows), **{k: mean(rows, k) for k in fields}})
        for name in ['DONE.json', 'metrics.json', 'geometry.json', 'protocol.json', 'receipts.json']:
            path = directory / name
            receipts[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for row in geos:
        if row['condition'] in ['intact', 'values_only']:
            assert row['attention_tv_intact'] == 0
    return {'cells': cells, 'geometry_means': geos,
            'frozen_prediction_sha256': hashlib.sha256((discovery / 'prediction.json').read_bytes()).hexdigest(),
            'prediction_assessment': assessments, 'source_sha256': receipts,
            'scope': 'New K/V outcomes tested prospectively; 50k intact/removal endpoints already known. No new model or target.'}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data-root', type=Path, default=Path(__file__).parent / 'data')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--summary-output', type=Path, required=True)
    args = p.parse_args()
    result = summarize(args.data_root)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelsize': 8,
        'axes.titlesize': 9, 'axes.titleweight': 'bold', 'xtick.labelsize': 7.2, 'ytick.labelsize': 7})
    fig = plt.figure(figsize=(8, 4.75), facecolor='white')
    heat = fig.add_axes([.17, .23, .285, .67])
    effects = fig.add_axes([.565, .55, .405, .35])
    internal = fig.add_axes([.565, .15, .405, .23])

    with (args.data_root / 'message_scale_20260907/nine_source_map.csv').open() as f:
        grid = list(csv.DictReader(f))
    values = np.empty((9, 5))
    for yi, source in enumerate(SOURCES):
        for xi, target in enumerate(TARGETS):
            pair = [float(r['support_delta_points']) for r in grid if r['source'] == source and r['target'] == target]
            assert len(pair) == 2
            values[yi, xi] = np.mean(pair)
    heat.imshow(values, aspect='auto', cmap='RdBu', norm=TwoSlopeNorm(vmin=-11, vcenter=0, vmax=11))
    for yi, source in enumerate(SOURCES):
        for xi, target in enumerate(TARGETS):
            value = values[yi, xi]
            pair = [float(r['support_delta_points']) for r in grid if r['source'] == source and r['target'] == target]
            mixed = min(pair) < 0 < max(pair)
            heat.text(xi, yi, f'{value:+.1f}' + ('*' if mixed else ''), ha='center', va='center',
                      fontsize=7, color='white' if abs(value) > 6 else '#192B3C')
            if source == target:
                heat.add_patch(Rectangle((xi-.48, yi-.48), .96, .96, fill=False, lw=.8, color='#192B3C'))
    heat.set_xticks(range(5), ['CP', 'E20', 'FB', 'BOT', 'SUSP'])
    heat.set_yticks(range(9), SOURCE_LABELS)
    heat.tick_params(length=0)
    for spine in heat.spines.values(): spine.set_visible(False)
    heat.set_title('A  Broader role-intervention evidence', loc='left', pad=11)
    fig.text(.17, .155, 'Support-removal pooled AUC points\nBlue helps; red hurts; * signs differ.', fontsize=7, linespacing=1.3)
    fig.text(.17, .085, 'Two streams; one historical checkpoint/source.\nBox: source = target. This is not the K/V test.', fontsize=6.6, linespacing=1.3)

    groups = [('Discovery: 2.5k', 'original'), ('Prediction test: 50k', 'original'), ('Prediction test: 50k', 'fresh')]
    for ci, condition in enumerate(CONDITIONS):
        ys = []
        for config, stream in groups:
            found = [r for r in result['cells'] if r['configuration'] == config and r['stream'] == stream and r['condition'] == condition]
            assert len(found) == 1
            ys.append(found[0]['delta_within_points'])
        xs = np.arange(3) + (ci - 1)*.235
        effects.bar(xs, ys, width=.21, color=COLORS[condition])
        for x, y in zip(xs, ys):
            effects.text(x, y+.22, f'{y:.2f}', ha='center', fontsize=6.8)
    effects.set_ylim(0, 12.2)
    effects.set_xticks(range(3), ['2.5k discovery', '50k original', '50k fresh'])
    effects.set_ylabel('Within-episode AUC gain (points)')
    effects.axvline(.5, color='#9BA4AC', lw=.8, ls='--')
    effects.grid(axis='y', alpha=.13)
    effects.set_axisbelow(True)
    effects.set_title('B  The value-path prediction survives', loc='left', pad=11)
    handles = [Line2D([], [], marker='s', color=COLORS[c], ls='', ms=5, label=LABELS[c]) for c in CONDITIONS]
    effects.legend(handles=handles, ncol=3, loc='upper center', bbox_to_anchor=(.5, -.14),
                   frameon=False, fontsize=6.8, handletextpad=.3, columnspacing=.65)

    internal.set_title('C  Routing changes; values move the contrast', loc='left', pad=12)
    for config, marker, size in [('Discovery: 2.5k', 'o', 28), ('Prediction test: 50k', 's', 35)]:
        for condition in CONDITIONS:
            row = next(r for r in result['geometry_means'] if r['configuration'] == config and r['condition'] == condition)
            x = row['attention_tv_intact']; y = 1-row['final_orientation_cos_intact']
            internal.scatter(x, y, marker=marker, s=size, color=COLORS[condition], zorder=3)
    internal.set_xlim(-.01, .18); internal.set_ylim(-.045, .82)
    internal.set_xticks([0, .05, .10, .15])
    internal.set_yticks([0, .4, .8])
    internal.set_xlabel('Attention change from intact (total variation)', labelpad=2)
    internal.set_ylabel('Contrast rotation\n(1 - cosine)', labelpad=2)
    internal.grid(alpha=.13)
    internal.text(.095, .64, 'Values only:\nattention exactly fixed', fontsize=6.8,
                  ha='center', color=COLORS['values_only'])
    internal.text(.105, .27, 'Keys only', fontsize=6.8, color=COLORS['keys_only'])
    internal.annotate('', xy=(.146, .045), xytext=(.137, .255),
                      arrowprops={'arrowstyle': '-', 'color': COLORS['keys_only'], 'lw': .7})
    fig.text(.565, .015, 'C: episode means; circles = 2.5k, squares = 50k (both streams).\nB: 128 episodes/cell; one checkpoint/configuration. No seed-level CI.', fontsize=6.6, linespacing=1.3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=260, facecolor='white')
    plt.close(fig)
    args.summary_output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'figure': str(args.output), 'summary': str(args.summary_output),
                      'prediction_passed_both_streams': True}, indent=2))


if __name__ == '__main__':
    main()
