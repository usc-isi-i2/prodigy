"""Sequential discovery/prediction test of support attention versus value paths.

One initial HK political cell discovers a component ordering; the same fixed
50k checkpoint tests that ordering on its two existing political streams.
No new training, target search, query fitting, or operator repair.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments.run_shared_graph import write_json
from .analyze_mixture_predictions import evaluate_logits
from .class_reference_kv import kv_forward
from .contrast_metrics import ranking_metrics
from .replay import batch_hash
from .run_class_reference_long import recorded_model
from .run_episode_cardinality import make_model
from .verify_member_training import model_digest


TARGET = 'covid_political'
CONDITIONS = ['intact', 'removed', 'keys_only', 'values_only', 'joint_kv']


def prediction_rule(rows):
    """One-point practical discovery separation, never a significance claim."""
    scores = {r['condition']: r['within_episode_auc'] for r in rows}
    d = {k: scores[k] - scores['intact'] for k in ['keys_only', 'values_only', 'joint_kv']}
    chosen = None
    for name, other in [('keys_only', 'values_only'), ('values_only', 'keys_only')]:
        if d[name] > 0 and d[name] - d[other] >= .01:
            chosen = name
    if chosen is None and d['keys_only'] <= 0 and d['values_only'] <= 0 and d['joint_kv'] > 0:
        chosen = 'joint_dependence'
    return {'hypothesis': chosen or 'ambiguous', 'discovery_within_deltas': d,
        'rule': 'Select positive singleton exceeding the other by at least 1 AUC point; otherwise joint dependence only when both singletons are nonpositive and joint positive; else stop.',
        'prediction': 'Selected singleton improves within-episode AUC and exceeds the other in EACH 50k stream; joint dependence predicts positive joint and nonpositive singletons in each stream. No pooled substitution.',
        'practical_separation_auc': .01, 'not_statistical_significance': True}


def load_cell(args, stream):
    read = lambda path: json.loads(path.read_text())
    if args.phase == 'discovery':
        cp = read(args.contrast_root / 'protocol.json')
        scale = Path(cp['scale_root']); dose = Path(cp['dose_root'])
        top = read(Path(read(scale / 'protocol.json')['reference_replay']) / 'protocol.json')
        recs = [r for r in read(dose / 'checkpoint_inventory.json') if
                r['source'] == 'cp_hk' and r['seed'] == 0 and r['step'] == 2500]
        if len(recs) != 1: raise ValueError('discovery checkpoint not unique')
        rec = recs[0]
        saved = torch.load(scale / 'predictions' / TARGET / stream / (rec['model_id'] + '.pt'), map_location='cpu', weights_only=False)
        directory = Path(top['cache_roots'][TARGET + '/' + stream]) / TARGET
        paths = [directory / 'batches' / f'batch_{i:03d}.pt' for i in range(32)]
        labels = saved['labels']; expected = labels['cache']['batch_sha256']
        state = torch.load(rec['checkpoint'], map_location='cpu', weights_only=True)['model']
        first = torch.load(paths[0], map_location='cpu', weights_only=False)
        model = make_model(read(directory.parent / 'protocol.json'), TARGET, labels['cache']['graph_path'], first[0].x.shape[1], state)
        reference = [saved['logits'][k] for k in ['scale_1', 'scale_0']]
    else:
        cp = read(args.long_root / 'protocol.json')
        rec = {'checkpoint': cp['checkpoint'], 'weights_sha256': cp['weights_sha256'],
               'model_id': 'source_selected_hk_50000', 'seed': 0, 'step': 50000, 'source': 'cp_hk'}
        saved = torch.load(args.long_root / 'predictions' / TARGET / (stream + '.pt'), map_location='cpu', weights_only=False)
        labels = saved['labels']
        paths = [args.long_root / 'private_inputs' / TARGET / stream / f'batch_{i:03d}.pt' for i in range(128)]
        receipts = [r for r in read(args.long_root / 'receipts.json') if r['target'] == TARGET and r['stream'] == stream]
        if len(receipts) != 1: raise ValueError('50k input receipt not unique')
        expected = receipts[0]['batch_sha256']
        state = torch.load(rec['checkpoint'], map_location='cpu', weights_only=True)['model']
        params = {**cp['recorded_training_params'], 'task_name': 'classification',
                  'ignore_label_embeddings': False, 'zero_shot': False, 'n_way': 2, 'n_shots': 3, 'batch_size': 1}
        model = recorded_model(params, state)
        reference = [saved['logits'][k] for k in ['native_intact', 'native_removed']]
    if model_digest(model.state_dict()) != rec['weights_sha256']:
        raise ValueError('selected weights changed')
    if len(paths) != len(expected) or len(labels['batch_counts']) != len(paths):
        raise ValueError('incomplete original input batches')
    return model, labels, paths, expected, reference, rec


def geometry(audit, intact, removed, batch):
    q = audit['query_mask']; n = len(q); edges = audit['edge_index']
    labels = audit['final_labels'].double(); a = intact['final_labels'].double(); b = removed['final_labels'].double()
    cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x[None], y[None], dim=1))
    ids = batch[0].task_id_per_sample
    right = batch[3][1].reshape(-1, 2)
    rows = []
    for episode in ids[q].unique().tolist():
        qi = torch.where(q & (ids == episode))[0][0]
        pair = right[qi] - n
        z, za, zb = (x[pair] for x in [labels, a, b])
        center = z.mean(0); half = (z[1] - z[0]) / 2
        ch = center / center.norm(); tangent = half - (half * ch).sum() * ch
        normed = lambda x: x / x.norm(dim=1, keepdim=True)
        d, da, db = [(normed(x)[1] - normed(x)[0]) for x in [z, za, zb]]
        to_pair = (edges[1, :, None] == (pair + n)[None]).any(1)
        weights = audit['attention'].double()
        # Each class/head attention distribution sums to one; mean TV over 2 classes.
        tv = lambda donor: float((weights[to_pair] - donor['attention'][to_pair].double()).abs().sum() / (4 * weights.shape[1]))
        mass = {}
        for name in ['label_positive', 'label_negative', 'label_self']:
            take = to_pair & audit['edge_roles'][name]
            mass[name + '_mass'] = float(weights[take].sum() / (2 * weights.shape[1]))
        rows.append({'episode_in_batch': int(episode), 'label_center_norm': float(center.norm()),
            'label_half_difference_norm': float(half.norm()), 'tangent_fraction': float(tangent.norm() / half.norm().clamp_min(1e-12)),
            'final_contrast_strength': float(d.norm()), 'center_cos_intact': cos(center, za.mean(0)),
            'unnormalized_difference_cos_intact': cos(half, (za[1] - za[0]) / 2),
            'final_orientation_cos_intact': cos(d, da), 'final_orientation_cos_removed': cos(d, db),
            'attention_tv_intact': tv(intact), 'attention_tv_removed': tv(removed), **mass})
    return rows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--phase', choices=['discovery', 'long_test'], required=True)
    p.add_argument('--contrast-root', type=Path, required=True)
    p.add_argument('--long-root', type=Path, required=True)
    p.add_argument('--discovery-root', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available(): raise ValueError('new output and hidden GPUs required')
    torch.set_num_threads(4)
    for root, count, key in [(args.contrast_root, 252, 'score_cells'), (args.long_root, 84, 'cells')]:
        done = json.loads((root / 'DONE.json').read_text())
        if done[key] != count or done['smoke_only']: raise ValueError('complete parent experiments required')
    prediction = None
    if args.phase == 'long_test':
        if args.discovery_root is None: raise ValueError('frozen discovery prediction required')
        if not (args.discovery_root / 'DONE.json').exists(): raise ValueError('discovery incomplete')
        prediction = json.loads((args.discovery_root / 'prediction.json').read_text())
        if prediction['hypothesis'] == 'ambiguous': raise ValueError('ambiguous discovery: do not force a follow-up')
    protocol = {'phase': args.phase, 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'target': TARGET, 'source': 'cp_hk', 'streams': ['original'] if args.phase == 'discovery' else ['original', 'fresh'],
        'conditions': CONDITIONS, 'prediction_rule': prediction_rule.__doc__, 'new_training': False,
        'discovery_selection': 'Initial HK seed0 step2500, political original128 episodes, selected before new K/V outcomes.',
        'frozen_prediction': prediction, 'primary_metric': 'within_episode_auc',
        'interpretation': 'Transplant actual support K/V only. Joint endpoint must reproduce removal and queries remain fixed. Component ordering is not a variance partition or representativeness claim.',
        'discovery_rule_details': {'practical_separation_auc': .01,
            'component': 'Positive singleton exceeds other singleton by at least .01 within-AUC.',
            'joint': 'Both singleton deltas nonpositive while joint delta positive.',
            'otherwise': 'Ambiguous; stop without testing a forced prediction.'}}
    args.output.mkdir(parents=True); write_json(args.output / 'protocol.json', protocol)
    all_metrics = []; all_geo = []; receipts = []; start = time.monotonic()
    for stream in protocol['streams']:
        model, labels, paths, expected, reference, rec = load_cell(args, stream)
        values = {c: [] for c in CONDITIONS}; checks = 0; max_error = 0.
        for bi, path in enumerate(paths):
            batch = torch.load(path, map_location='cpu', weights_only=False)
            if batch_hash(batch) != expected[bi]: raise ValueError('input batch changed')
            a = kv_forward(model, batch, alpha=1.); b = kv_forward(model, batch, alpha=0.)
            donor = b[4]['kqv_native']; saved_donor = donor.clone()
            outcomes = {'intact': a, 'removed': b,
                'keys_only': kv_forward(model, batch, key_donor=donor),
                'values_only': kv_forward(model, batch, value_donor=donor),
                'joint_kv': kv_forward(model, batch, key_donor=donor, value_donor=donor)}
            n = len(a[4]['query_mask']); to_label = a[4]['edge_index'][1] >= n
            q = a[4]['query_mask']
            for condition, result in outcomes.items():
                logits, trace, parts, error, audit = result
                values[condition].append(logits)
                torch.testing.assert_close(audit['final_inputs'][q], a[4]['final_inputs'][q], rtol=0, atol=0); checks += 1
                expected_attention = b[4] if condition in ['removed', 'keys_only', 'joint_kv'] else a[4]
                torch.testing.assert_close(audit['attention'][to_label], expected_attention['attention'][to_label], rtol=0, atol=0)
                all_geo.extend({'stream': stream, 'condition': condition, 'batch': bi, **row}
                               for row in geometry(audit, a[4], b[4], batch))
                max_error = max(max_error, error)
            torch.testing.assert_close(outcomes['joint_kv'][0], b[0], rtol=0, atol=0)
            torch.testing.assert_close(outcomes['joint_kv'][4]['final_labels'], b[4]['final_labels'], rtol=0, atol=0)
            torch.testing.assert_close(donor, saved_donor, rtol=0, atol=0)
            if batch_hash(batch) != expected[bi]: raise ValueError('input mutated')
            private = args.output / 'private_activations' / stream; private.mkdir(parents=True, exist_ok=True)
            torch.save({c: r[4] for c, r in outcomes.items()}, private / f'batch_{bi:03d}.pt')
        common = {'source': 'cp_hk', 'target': TARGET, 'stream': stream, 'step': rec['step'], 'seed': 0}
        logits = {k: torch.cat(v) for k, v in values.items()}
        for k, ref in zip(['intact', 'removed'], reference):
            torch.testing.assert_close(logits[k], ref, rtol=0, atol=0)
        for condition, value in logits.items():
            ranks, _ = ranking_metrics((value[:, 1].double() - value[:, 0].double()).numpy(),
                labels['local_y'].numpy(), labels['mapping'].numpy(), labels['episode_ids'].numpy(), labels['use_global'])
            all_metrics.append({**common, 'condition': condition, **evaluate_logits(value, labels), **ranks})
        if model_digest(model.state_dict()) != rec['weights_sha256']: raise ValueError('weights changed')
        receipts.append({**common, **rec, 'batches': len(paths), 'batch_sha256': expected,
            'exact_query_checks': checks, 'native_endpoints_bit_exact': True, 'joint_endpoint_bit_exact': True,
            'label_attention_routes_bit_exact': True, 'max_decoder_margin_error': max_error})
        private = args.output / 'private_predictions'; private.mkdir(exist_ok=True)
        torch.save({'logits': logits, 'labels': labels}, private / (stream + '.pt'))
        for name, value in [('metrics', all_metrics), ('geometry', all_geo), ('receipts', receipts)]:
            write_json(args.output / (name + '.json'), value)
        print(json.dumps({'stream': stream, 'cells': len(all_metrics), 'elapsed': time.monotonic() - start}), flush=True)
    if args.phase == 'discovery':
        prediction = prediction_rule(all_metrics)
        write_json(args.output / 'prediction.json', prediction)
    else:
        assessment = []
        for stream in protocol['streams']:
            sub = {r['condition']: r['within_episode_auc'] for r in all_metrics if r['stream'] == stream}
            d = {k: sub[k] - sub['intact'] for k in CONDITIONS}
            h = prediction['hypothesis']
            passed = (d['joint_kv'] > 0 and d['keys_only'] <= 0 and d['values_only'] <= 0) if h == 'joint_dependence' else (
                d[h] > 0 and d[h] > d['values_only' if h == 'keys_only' else 'keys_only'])
            assessment.append({'stream': stream, 'within_deltas': d, 'primary_prediction_passed': passed})
        write_json(args.output / 'prediction_assessment.json', assessment)
    write_json(args.output / 'DONE.json', {'phase': args.phase, 'cells': len(all_metrics), 'streams': len(receipts),
        'all_native_endpoints_bit_exact': True, 'all_joint_endpoints_bit_exact': True,
        'exact_query_checks': sum(r['exact_query_checks'] for r in receipts), 'new_training': False,
        'elapsed_seconds': time.monotonic() - start})


if __name__ == '__main__':
    main()
