"""Bounded frozen HK K/V study using saved embeddings only; CPU by default."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

os.environ.setdefault('WANDB_MODE', 'disabled')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.experiments.setup.nm_hk_support_extremes.run import MetadataOnly
from scripts.experiments.setup.nm_hk_mechanism.run import build_model, metrics
from scripts.experiments.setup.nm_support_resampling.run import digest_state
from scripts.experiments.setup.nm_hk_goal.metagraph import replay, compact_trace


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bank_metadata(path):
    # Read only candidate embeddings, logits and metadata. Do not materialize
    # the cached candidate graph features or import another full graph.
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        with archive.open(name) as stream:
            meta = MetadataOnly(stream).load()
        for field in ['embeddings', 'logits']:
            ref = meta[field]
            assert ref['storage']['type'] == 'FloatStorage'
            raw = archive.read(name[:-len('data.pkl')] + 'data/' + ref['storage']['key'])
            value = np.ndarray(ref['size'], dtype='<f4', buffer=raw, offset=4 * ref['offset'],
                strides=tuple(4 * stride for stride in ref['stride']))
            meta[field] = torch.from_numpy(value.copy())
        return {k: meta[k] for k in ['embeddings', 'logits', 'selections', 'logit_row_keys', 'head_order']}


def gpu_guard(device, sessions):
    if device == 'cpu':
        return
    physical = int(device.split(':')[1])
    assert physical in range(4)
    for session in sessions:
        if subprocess.run(['tmux', 'has-session', '-t', session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            raise RuntimeError('Priority session active: ' + session)
    output = subprocess.check_output(['nvidia-smi', '-i', str(physical), '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    if any(line.strip() and int(line.strip()) != os.getpid() for line in output.splitlines()):
        raise RuntimeError('Another process has priority on the requested GPU')


def parity(logits, reference, truth):
    probability_error = abs(float(logits.softmax(0)[truth]) - float(reference.native_probability))
    old = int(reference.native_prediction)
    shortfall = float(logits.max() - logits[old])
    assert probability_error < 1e-5 and shortfall < 1e-4, (probability_error, shortfall)
    return dict(probability_error=probability_error, winner_shortfall=shortfall,
        prediction_difference=int(logits.argmax()) != old,
        correctness_difference=bool(logits.argmax() == truth) != bool(reference.native_correct))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, default=Path('/dataMeR1/phil/gfm/error_audit/nm_hk_mechanism_20260908'))
    p.add_argument('--extremes', type=Path, default=Path('/dataMeR1/phil/gfm/error_audit/nm_hk_support_extremes_20260908'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--methods', nargs='+', default=['nearest'])
    p.add_argument('--priority-sessions', nargs='*', default=['paper-three-seed-fast', 'paper-flagship-wait'])
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    assert 1 <= a.threads <= 2
    if a.dry_run:
        print(json.dumps(dict(device=a.device, threads=a.threads, cases=200, methods=a.methods,
            graph_loading=False, encoding=False, training=False, priority_sessions=a.priority_sessions), indent=2))
        return
    gpu_guard(a.device, a.priority_sessions)
    a.out.mkdir(parents=True, exist_ok=False)
    os.nice(10)
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    started = time.time()
    receipts = {k: json.loads((root / 'receipt.json').read_text()) for k, root in [('original', a.original), ('extremes', a.extremes)]}
    original_path = a.original / 'embeddings_and_logits_private.pt'
    bank_path = a.extremes / 'candidate_bank_private.pt'
    assert digest(original_path) == receipts['original']['cache_sha256']
    assert digest(bank_path) == receipts['extremes']['candidate_bank_sha256']
    cached = torch.load(original_path, map_location='cpu', weights_only=False)
    bank = bank_metadata(bank_path)
    params = json.loads((a.original / 'effective_config.json').read_text())
    params['device'] = a.device
    model = build_model(params, cached['checkpoint'], a.device)
    before = digest_state(model)
    assert before == cached['model_state_sha256'] == receipts['extremes']['model_state_sha256']
    old_table = pd.read_csv(a.original / 'head_comparison_private.csv')
    old = old_table[old_table.condition.eq('original')].set_index('case')
    selected = pd.read_csv(a.extremes / 'selection_results_private.csv').set_index(['case', 'method', 'draw'])
    assert digest(a.original / 'head_comparison_private.csv') == receipts['original']['csv_sha256']
    assert digest(a.extremes / 'selection_results_private.csv') == receipts['extremes']['csv_sha256']
    selectors = {(int(v['case']), v['method'], int(v['draw'])): v for v in bank['selections']}
    results, traces, audits = [], {}, []
    for ci, c in enumerate(cached['cases']):
        gpu_guard(a.device, a.priority_sessions)
        args = tuple(v.to(a.device).clone() for v in cached['episodes'][c['batch'], c['ep']])
        original_copies = [v.clone() for v in args]
        slots = [c['truth'] * 7 + j for j in range(3)]
        base, base_a = replay(model, args)
        audits.append(dict(case=c['case'], endpoint='original', **parity(base[c['local']], old.loc[c['case']], c['truth'])))
        for method in a.methods:
            choices = [key for key in selectors if key[0] == c['case'] and key[1] == method]
            for key in sorted(choices):
                selection = selectors[key]
                new = list(args)
                new[0] = args[0].clone()
                new[0][slots] = bank['embeddings'][selection['embedding_indices']].to(a.device)
                full, full_a = replay(model, new)
                audits.append(dict(case=c['case'], endpoint=str(key), **parity(full[c['local']], selected.loc[key], c['truth'])))
                assert torch.equal(full_a['queries'], base_a['queries'])
                donor = full_a['native_kqv']
                runs = [('original', base, base_a), ('full', full, full_a)]
                for label, k, v in [('keys', donor, None), ('values', None, donor), ('joint', donor, donor)]:
                    z, capture = replay(model, args, changed_rows=slots, key_donor=k, value_donor=v)
                    assert torch.equal(capture['queries'], base_a['queries'])
                    label_edges = capture['edges'][1] >= len(args[0])
                    if label == 'values':
                        assert torch.equal(capture['attention'], base_a['attention'])
                    else:
                        assert torch.equal(capture['attention'][label_edges], full_a['attention'][label_edges])
                    if label == 'joint':
                        assert float((capture['labels'] - full_a['labels']).abs().max()) < 1e-5
                        assert float((z[capture['query_mask']] - full[capture['query_mask']]).abs().max()) < 1e-4
                    runs.append((label, z, capture))
                for label, z, capture in runs:
                    trace = compact_trace(capture, c['local'], c['truth'], float(model.logit_scale.exp()))
                    assert float((sum(trace['terms'].values()) - z[c['local']].cpu()).abs().max()) < 1e-4
                    row = dict(case=c['case'], cohort=c['cohort'], method=method, draw=key[2], condition=label,
                        persistent=bool(selected.loc[key, 'previously_persistent']),
                        original_correct=int(old.loc[c['case'], 'native_correct']))
                    row.update(metrics(z[c['local']], c['truth'], 'native'))
                    for part, values in trace['terms'].items():
                        row['true_score_' + part] = float(values[c['truth']])
                    row['true_positive_attention'] = float(trace['attention_mass']['positive'][c['truth']].mean())
                    row['true_negative_attention'] = float(trace['attention_mass']['negative'][c['truth']].mean())
                    row['true_label_norm'] = float(trace['labels'][c['truth']].norm())
                    row['max_accounting_error'] = trace['max_accounting_error']
                    results.append(row)
                    traces[c['case'], method, key[2], label] = trace
        for old_arg, new_arg in zip(original_copies, args):
            assert torch.equal(old_arg, new_arg)
        if (ci + 1) % 20 == 0:
            print('CASES', ci + 1, 'seconds', round(time.time() - started, 2), flush=True)
    assert digest_state(model) == before
    frame = pd.DataFrame(results)
    assert not frame.duplicated(['case', 'method', 'draw', 'condition']).any()
    frame.to_csv(a.out / 'results_private.csv', index=False)
    torch.save(dict(traces=traces, cases=cached['cases'], model_state_sha256=before), a.out / 'traces_private.pt')
    report = dict(complete=True, seconds=time.time() - started, device=a.device, threads=a.threads,
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        original_cache_sha256=receipts['original']['cache_sha256'], bank_sha256=receipts['extremes']['candidate_bank_sha256'],
        model_state_sha256=before, unchanged_model=True, methods=a.methods, rows=len(frame),
        csv_sha256=digest(a.out / 'results_private.csv'), traces_sha256=digest(a.out / 'traces_private.pt'),
        parity=audits, graph_loading=False, encoding=False, training=False)
    (a.out / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print('COMPLETE', report['seconds'], flush=True)


if __name__ == '__main__':
    main()
