"""Read saved binary logits: distinguish class preference from discrimination.

No model forwards or fitted correction. Centering is descriptive and uses the
observed query cohort, not a proposed deployment rule. Export only aggregates
and first-episode anonymous numeric cases; retain all private inputs on host.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
import torch


CONDITIONS = ['native_intact', 'native_removed', 'support_direction_only', 'raw_prototype']
TARGETS = ['covid_political', 'facebook_page_reference', 'twibot20']


def decompose(margin, y, ids, predictions):
    margin = np.asarray(margin, dtype=np.float64)
    offsets = np.zeros_like(margin)
    within = []; constant_decisions = 0
    for episode in np.unique(ids):
        take = ids == episode
        offsets[take] = margin[take].mean()
        within.append(roc_auc_score(y[take], margin[take]))
        constant_decisions += len(np.unique(predictions[take])) == 1
    centered = margin - offsets
    error = float(np.max(np.abs(margin - offsets - centered)))
    centered_within = [roc_auc_score(y[ids == e], centered[ids == e]) for e in np.unique(ids)]
    if not np.allclose(within, centered_within, rtol=0, atol=1e-12):
        raise ValueError('Episode translation changed ranking beyond rounding.')
    residual_sd = float(centered.std())
    if residual_sd <= 0:
        raise ValueError('No within-episode score variation.')
    result = {'queries': len(y), 'episodes': len(within), 'positive_fraction': float(y.mean()),
        'positive_decision_fraction': float(predictions.mean()), 'accuracy': float(accuracy_score(y, predictions)),
        'pooled_margin_auc': float(roc_auc_score(y, margin)), 'within_episode_auc': float(np.mean(within)),
        'margin_mean': float(margin.mean()), 'margin_sd': float(margin.std()),
        'episode_mean_sd': float(np.std([offsets[ids == e][0] for e in np.unique(ids)])),
        'within_residual_sd': residual_sd, 'constant_decision_episodes': int(constant_decisions),
        'zero_margin_queries': int((margin == 0).sum()), 'decomposition_max_error': error,
        'episode_translation_preserves_ranking': True}
    for label, name in [(0, 'negative'), (1, 'positive')]:
        for values, mode in [(margin, 'native'), (centered, 'centered')]:
            sample = values[y == label]
            result[mode + '_' + name + '_mean'] = float(sample.mean())
            result[mode + '_' + name + '_quantiles'] = np.quantile(sample, [.05, .25, .5, .75, .95]).tolist()
    return result, centered, offsets, residual_sd


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--code-revision', required=True)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output folder.')
    torch.set_num_threads(2)
    completed = json.loads((args.input / 'DONE.json').read_text())
    if completed['cells'] != 84 or completed['smoke_only']:
        raise ValueError('Requires completed nominated long-checkpoint test.')
    native = json.loads((args.input / 'metrics.json').read_text())
    summaries = []; distributions = []; cases = []; receipts = []
    bins = np.linspace(-6, 6, 81)
    for target in TARGETS:
        for stream in ['original', 'fresh']:
            path = args.input / 'predictions' / target / (stream + '.pt')
            # This is our own completed-run artifact, including NumPy margins.
            saved = torch.load(path, map_location='cpu', weights_only=False)
            lab = saved['labels']; local_y = lab['local_y'].numpy(); mapping = lab['mapping'].numpy()
            ids = lab['episode_ids'].numpy(); rows = np.arange(len(local_y))
            y = mapping[rows, local_y] if lab['use_global'] else local_y
            sign = np.where(mapping[:, 1] == 1, 1., -1.) if lab['use_global'] else np.ones(len(y))
            for condition in CONDITIONS:
                logits = saved['logits'][condition]
                local_pred = logits.argmax(1).numpy()
                pred = mapping[rows, local_pred] if lab['use_global'] else local_pred
                margin = (logits[:, 1].double() - logits[:, 0].double()).numpy() * sign
                metrics, centered, offsets, scale = decompose(margin, y, ids, pred)
                ref = [r for r in native if r['target'] == target and r['stream'] == stream and
                       r['family'] == 'readout' and r['condition'] == condition]
                if len(ref) != 1:
                    raise ValueError('Native metric row is not unique.')
                for name in ['accuracy', 'pooled_margin_auc', 'within_episode_auc']:
                    if abs(metrics[name] - ref[0][name]) > 1e-10:
                        raise ValueError('Saved prediction result differs: ' + name)
                common = dict(target=target, stream=stream, condition=condition)
                summaries.append({**common, **metrics})
                for label in [0, 1]:
                    for name, values in [('native', margin), ('centered', centered)]:
                        z = values[y == label] / scale
                        counts, _ = np.histogram(z, bins=bins)
                        distributions.append({**common, 'label': label, 'mode': name, 'count': len(z),
                            'scale': scale, 'bins': bins.tolist(), 'counts': counts.tolist(),
                            'below': int((z < bins[0]).sum()), 'above': int((z > bins[-1]).sum())})
                first = ids == ids.min()
                cases.append({**common, 'selection': 'First saved episode, no outcome-based selection.',
                    'episode': int(ids.min()), 'true_labels': y[first].tolist(), 'predictions': pred[first].tolist(),
                    'margins': margin[first].tolist(), 'episode_mean': float(offsets[first][0]),
                    'centered_margins': centered[first].tolist()})
            receipts.append({'target': target, 'stream': stream, 'path': str(path),
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    args.output.mkdir(parents=True)
    for name, value in [('summary', summaries), ('distributions', distributions), ('numeric_cases', cases), ('receipts', receipts)]:
        (args.output / (name + '.json')).write_text(json.dumps(value, indent=2) + '\n')
    protocol = {'code_revision': args.code_revision, 'source_run': str(args.input), 'new_model_forwards': 0,
        'conditions': CONDITIONS, 'targets': TARGETS, 'streams': ['original', 'fresh'],
        'interpretation': 'Descriptive decomposition of existing margins into episode mean plus centered residual. No fitted threshold, deployment correction, or new intervention.',
        'normalization': 'Histograms divide by per-target/stream/condition within-episode residual SD; positive scaling preserves decisions and ranks.',
        'labels': 'Original global/local mappings and argmax tie-breaking retained. Labels only describe class-conditional distributions, not choose corrections.',
        'case_selection': 'First episode in each saved stream; illustrative numeric cases, not representative selected biographies.'}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    done = {'summary_cells': len(summaries), 'histograms': len(distributions), 'case_sets': len(cases),
            'verified_files': len(receipts), 'native_metrics_reproduced': True, 'new_model_forwards': 0,
            'episode_translation_preserves_ranking': True}
    (args.output / 'DONE.json').write_text(json.dumps(done, indent=2) + '\n')
    print(json.dumps(done, indent=2))


if __name__ == '__main__':
    main()
