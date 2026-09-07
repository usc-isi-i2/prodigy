"""Fixed, paired outcome audit of the eight encoder/solver training arms."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score

from scripts.experiments.setup.encoder_solver_isolation.eval_plan import TARGETS
from scripts.experiments.setup.target_performance_mechanisms.analyze_mixture_predictions import input_labels, evaluate_logits

DECODERS = ('full_model', 'U1_pre_meta/ridge', 'raw_center/ridge')
METRICS = ('accuracy', 'macro_f1', 'f1', 'roc_auc', 'nll')
STREAMS = ('original', 'fresh')


def metrics(logits, labels):
    result = evaluate_logits(logits, labels)
    truth, pred = labels['local_y'], logits.argmax(1)
    if labels['use_global']:
        rows = torch.arange(len(truth))
        truth = labels['mapping'][rows, truth]
        # Production takes argmax AFTER global probability mapping. Mapping a
        # local argmax is not equivalent at ties (including softmax rounding).
        probs = torch.softmax(logits, dim=1)
        global_probs = torch.zeros_like(probs)
        global_probs.scatter_add_(1, labels['mapping'], probs)
        pred = global_probs.argmax(1)
    result['accuracy'] = float(accuracy_score(truth.numpy(), pred.numpy()))
    result['f1'] = float(f1_score(truth.numpy(), pred.numpy(), zero_division=0))
    result['macro_f1'] = float(f1_score(truth.numpy(), pred.numpy(), labels=[0, 1], average='macro', zero_division=0))
    return result


def inventory(training):
    done = json.loads((training / 'DONE.json').read_text())
    plan = json.loads((training / 'plan.json').read_text())
    expected = {(s, m) for s in ('blocked', 'interleaved') for m in ('native', 'joint', 'isolated', 'ridge_only')}
    if done['steps'] != 2500 or done['smoke'] or done['arms'] != 8:
        raise ValueError('completed full eight-arm training required')
    if len(plan) != 8 or len({p['name'] for p in plan}) != 8 or {(p['arm']['schedule'], p['mode']) for p in plan} != expected:
        raise ValueError('eight unique schedule/objective models required')
    return plan


def summarize(cells):
    means, deltas = [], []
    for stream in STREAMS:
        for panel, targets in [('all5', set(TARGETS)), ('fixed4', set(TARGETS) - {'ukr_rus_suspended'})]:
            for schedule in ('blocked', 'interleaved'):
                selected = [r for r in cells if r['stream'] == stream and r['target'] in targets and r['schedule'] == schedule]
                for mode in ('native', 'joint', 'isolated', 'ridge_only'):
                    for decoder in DECODERS:
                        group = [r for r in selected if r['mode'] == mode and r['decoder'] == decoder]
                        if len(group) != len(targets) or {r['target'] for r in group} != targets:
                            raise ValueError('missing or duplicate target cells')
                        means.append(dict(stream=stream, panel=panel, schedule=schedule, mode=mode, decoder=decoder,
                                          **{k: float(np.mean([r[k] for r in group])) for k in METRICS}))
                indexed = {(r['target'], r['mode'], r['decoder']): r for r in selected}
                for name, mode, decoder in [('native', 'native', 'full_model'), ('joint', 'joint', 'full_model'), ('own_U1', 'isolated', 'U1_pre_meta/ridge')]:
                    paired = [{k: indexed[t, 'isolated', 'full_model'][k] - indexed[t, mode, decoder][k] for k in METRICS} for t in sorted(targets)]
                    deltas.append(dict(stream=stream, panel=panel, schedule=schedule, comparison='isolated_full_minus_' + name,
                                       **{k: float(np.mean([r[k] for r in paired])) for k in METRICS}))
    return means, deltas


def analyze(training, replay):
    plan = inventory(training)
    cells, receipts, weight_hashes = [], [], {}
    for stream in STREAMS:
        for target in TARGETS:
            root = replay / stream / target
            if not (root / 'DONE').is_file():
                raise ValueError(f'incomplete replay: {root}')
            directory = root / target
            labels = input_labels(directory, target)  # Checks all 32 cached tensor hashes.
            rows = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines()]
            if {r['model_id'] for r in rows} != {p['name'] for p in plan}:
                raise ValueError('replay model inventory differs from training plan')
            for entry in plan:
                name = entry['name']
                path = directory / (name + '__baseline.pt')
                records = torch.load(path, map_location='cpu', weights_only=False)
                if len(records) != 32:
                    raise ValueError('incomplete prediction export')
                predictions = {d: [] for d in DECODERS}
                for index, record in enumerate(records):
                    if record['batch'] != index or record['batch_sha256'] != labels['cache']['batch_sha256'][index]:
                        raise ValueError('prediction batch order/hash mismatch')
                    for decoder in DECODERS:
                        value = record['logits'][decoder]
                        if value.shape != (labels['batch_counts'][index], 2) or not torch.isfinite(value).all():
                            raise ValueError('invalid prediction shape or values')
                        predictions[decoder].append(value)
                for decoder in DECODERS:
                    matched = [r for r in rows if r['model_id'] == name and r['decoder'] == decoder and r['variant'] == 'baseline']
                    if len(matched) != 1:
                        raise ValueError('metric row not unique')
                    row = matched[0]
                    expected_checkpoint = training / 'state' / (name + '_isolation_v1') / 'checkpoint' / 'state_dict_2500.ckpt'
                    if (row['dataset'] != target or Path(row['checkpoint']) != expected_checkpoint or
                            sorted(row['sources']) != sorted(entry['arm']['sources']) or
                            row['episode_fingerprint'] != labels['cache']['episode_fingerprint'] or
                            row['queries'] != len(labels['local_y']) or row['episodes'] != 128):
                        raise ValueError('model/input provenance mismatch')
                    digest = row['weights_sha256']
                    if len(digest) != 64 or weight_hashes.setdefault(name, digest) != digest:
                        raise ValueError('model weights changed between cells')
                    result = metrics(torch.cat(predictions[decoder]), labels)
                    if any(abs(result[k] - row[k]) > 1e-6 for k in ('accuracy', 'f1', 'roc_auc', 'nll')):
                        raise ValueError('logits do not reproduce saved metrics')
                    cells.append(dict(stream=stream, target=target, model_id=name, schedule=entry['arm']['schedule'],
                                      mode=entry['mode'], decoder=decoder, queries=len(labels['local_y']),
                                      f1_class_space='semantic_binary_positive_1' if labels['use_global'] else 'episode_local_positive_1', **result))
                receipts.append(dict(stream=stream, target=target, model_id=name, weights_sha256=digest,
                                     prediction_path=str(path), prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                     episode_fingerprint=labels['cache']['episode_fingerprint']))
    means, deltas = summarize(cells)
    indexed = {(r['stream'], r['target'], r['schedule'], r['mode'], r['decoder']): r for r in cells}
    paired_cells = []
    for stream in STREAMS:
        for target in TARGETS:
            for schedule in ('blocked', 'interleaved'):
                own = indexed[stream, target, schedule, 'isolated', 'full_model']
                for name, mode, decoder in [('native', 'native', 'full_model'), ('joint', 'joint', 'full_model'), ('own_U1', 'isolated', 'U1_pre_meta/ridge')]:
                    other = indexed[stream, target, schedule, mode, decoder]
                    paired_cells.append(dict(stream=stream, target=target, schedule=schedule,
                                             comparison='isolated_full_minus_' + name,
                                             **{k: own[k] - other[k] for k in METRICS}))
    return dict(cells=cells, means=means, deltas=deltas, paired_cells=paired_cells, receipts=receipts,
                notes=['Equal target weights; streams and schedules remain separate. Deltas are percentage fractions, not percentage points; lower NLL is better.',
                       'FB F1 and macro F1 refer to episode-local binary class assignments, not a fixed semantic positive class or macro F1 over all FB categories.',
                       'Other targets use global semantic binary labels. Recurring accounts and crossed cells are not independent replicates.',
                       'Fixed argmax decisions; no fitted thresholds, selection, confidence intervals, or causal claims.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = analyze(args.training.resolve(), args.replay.resolve())
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
