"""Source-selected parameter interpolation of existing seed-2 continuations."""
import argparse
import csv
import math
import shutil
import subprocess
from pathlib import Path

import torch

from . import async_convergence as ac
from . import async_extension as ex
from . import async_seed2_control as control
from .evaluation_artifacts import atomic_json
from .mini_transfer import identity

ALPHAS = tuple(i / 10 for i in range(11))
FLOORS = dict(zip(ac.SOURCES, (.9896764585971831, .9531494824886322)))
RULE = 'both own-singleton validation AUC floors; max Ukraine AUC, then Facebook AUC, then lower alpha'


def blend_states(kd, ukraine, alpha):
    if not math.isfinite(alpha) or not 0 <= alpha <= 1 or kd.keys() != ukraine.keys():
        raise ValueError('invalid alpha or incompatible state keys')
    result = {}
    for key, left in kd.items():
        right = ukraine[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f'incompatible model tensor: {key}')
        if left.is_floating_point():
            if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                raise ValueError(f'nonfinite endpoint parameter: {key}')
            result[key] = left.clone() if alpha == 0 else right.clone() if alpha == 1 else (1 - alpha) * left + alpha * right
        else:
            if not torch.equal(left, right):
                raise ValueError(f'different nonfloating buffer: {key}')
            result[key] = left.clone()
    return result


def select_merge(rows, floors):
    for row in rows:
        if any(not math.isfinite(v['auc']) for v in row['validation']):
            raise ValueError('nonfinite validation AUC')
    eligible = [r for r in rows if all(r['validation'][i]['auc'] >= floors[s] for i, s in enumerate(ac.SOURCES))]
    selected = max(eligible, key=lambda r: (r['validation'][0]['auc'], r['validation'][1]['auc'], -r['alpha'])) if eligible else None
    return selected, [r['alpha'] for r in eligible]


def inputs(args):
    root = Path(args.control_root)
    data = control.read(root / 'analysis_data.json')
    if control.read(root / 'results/COMPLETE.json')['status'] != 'complete':
        raise ValueError('control experiment is incomplete')
    if (data['manifest']['singleton_source_auc'] != FLOORS
            or not data['manifest']['audit']['start_model_adam_sampler_rng_identical']):
        raise ValueError('original source floors or shared-parent audit changed')
    summaries = [data['summary'][arm] for arm in ('kd_w010', 'ukraine_only')]
    for summary in summaries:
        if (summary['status'] != 'complete' or summary['seed'] != 2 or summary['data_seed'] != 0
                or summary['probe_seed'] != 0 or summary['additional_steps'] != 60000):
            raise ValueError('expected seed2 fixed +60k endpoints on data/probe seed0')
    for key in ('start_sha256', 'start_counts', 'graph_split_receipts', 'probe_sha256'):
        if summaries[0][key] != summaries[1][key]:
            raise ValueError(f'endpoints do not share {key}')
    if summaries[0]['protocol']['kd_weight'] != .1 or summaries[1]['protocol']['schedule'] != 'ukraine_only':
        raise ValueError('incorrect endpoint recipes')
    models = {m['run_id']: m for m in data['manifest']['models']}
    endpoints = []
    for alias, arm in (('kd_w010_fixed', 'kd_w010'), ('ukraine_updates_fixed', 'ukraine_only')):
        entry = models[alias]
        row = next(r for r in data['history'][arm] if r['additional_step'] == 60000)
        if row['counts'] != entry['counts'] or row['validation'] != entry['validation']:
            raise ValueError('frozen endpoint and source history disagree')
        if ex.sha256(row['checkpoint']) != entry['checkpoint_sha256']:
            raise ValueError('endpoint checkpoint changed')
        endpoints.append(dict(alias=alias, row=row, sha256=entry['checkpoint_sha256']))
    return data, summaries, endpoints


@torch.no_grad()
def validate(args, device):
    root = Path(args.root)
    if (root / 'candidates').exists() or (root / 'analysis_data.json').exists():
        raise FileExistsError('merge output already exists')
    data, summaries, endpoints = inputs(args)
    config = ac.load_config(args.config)
    provenance = ac.base.preflight(config)
    graphs = [ac.base.load_graph(s, config, 0, device) for s in ac.SOURCES]
    for s, g in zip(ac.SOURCES, graphs):
        if g['receipt'] != summaries[0]['graph_split_receipts'][s]:
            raise ValueError('source graph/split changed')
    validation = [ac.diagnostic.validation_pairs(g) for g in graphs]
    probes = []
    for s in ac.SOURCES:
        path = Path(summaries[0]['parent_root']) / 'node_neighbors/lp/async_kd' / f'probe_{s}.pt'
        if ex.sha256(path) != summaries[0]['probe_sha256'][s]:
            raise ValueError('original source probe changed')
        probes.append(ex.on_device(torch.load(path, map_location='cpu', weights_only=False), device))
    states = [control.checkpoint(e['row'])['model'] for e in endpoints]
    metadata = dict(training_seed=2, data_seed=0, new_optimizer_updates=0, inference_only=True,
                    formula='theta=(1-alpha)*KD0.1+alpha*Ukraine_only', alphas=ALPHAS,
                    endpoint_added_optimizer_updates=[60000, 60000], endpoints=endpoints,
                    singleton_source_auc=data['manifest']['singleton_source_auc'],
                    code_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    input_provenance=provenance, shared_parent_sha256=summaries[0]['start_sha256'])
    model = ac.base.BiasMLP('node_neighbors').to(device).eval()
    rows = []
    for i, alpha in enumerate(ALPHAS):
        merged = blend_states(*states, alpha)
        model.load_state_dict(merged)
        report = ac.measure(model, graphs, validation, probes)
        if i in (0, 10):
            expected = endpoints[0 if i == 0 else 1]['row']
            if any(report[key] != expected[key] for key in ('validation', 'training_probe')):
                raise ValueError('endpoint validation/probe measurements do not reproduce exactly')
        run_id = f'merge_{i:02d}'
        cp = root / 'candidates' / f'{run_id}.pt'
        ac.save(cp, dict(model=merged, step=0, inference_only=True, exact_sampling_resume=False,
                         metadata=dict(alpha=alpha, step_semantics='zero new optimizer updates', **metadata)))
        row = dict(alpha=alpha, run_id=run_id, checkpoint=str(cp), **report)
        rows.append(row)
        print(row, flush=True)
    selected, eligible = select_merge(rows, metadata['singleton_source_auc'])
    selection = dict(status='selected' if selected else 'no_feasible_merge', rule=RULE,
                     qualifying_alphas=eligible, selected_alpha=selected['alpha'] if selected else None,
                     candidate_run_id=selected['run_id'] if selected else None,
                     run_id='merge_selected' if selected else None)
    if selected:
        cp = root / 'node_neighbors/lp/merge_selected/best.pt'
        cp.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(selected['checkpoint'], cp)
        selection.update(checkpoint_identity=identity(cp), checkpoint_sha256=ex.sha256(cp),
                         validation=selected['validation'])
    metadata['endpoint_measurements_reproduced_exactly'] = True
    atomic_json(root / 'selection.json', selection)
    atomic_json(root / 'analysis_data.json', dict(metadata=metadata, rows=rows, selection=selection))


def evaluate(args):
    root = Path(args.root)
    selected = control.read(root / 'selection.json')
    if selected['status'] != 'selected':
        print('No qualifying merge; downstream evaluation skipped.')
        return
    cp = selected['checkpoint_identity']['path']
    if identity(cp) != selected['checkpoint_identity'] or ex.sha256(cp) != selected['checkpoint_sha256']:
        raise ValueError('selected weights changed before evaluation')
    data = control.read(root / 'analysis_data.json')
    if ac.base.preflight(ac.load_config(args.config)) != data['metadata']['input_provenance']:
        raise ValueError('fixed data changed after selection')
    torch.set_num_threads(4); torch.cuda.set_device(args.device); torch.backends.cuda.matmul.allow_tf32 = False
    args.seed = 0; args.views = ('node_neighbors',); args.targets = ac.base.SOURCES
    args.model_rows = [('merge_selected', 'merge_selected')]
    ac.base.evaluate(args, torch.device(f'cuda:{args.device}'))


def aggregate(args):
    root = Path(args.root)
    selection = control.read(root / 'selection.json')
    data = control.read(root / 'analysis_data.json')
    rows = []
    if selection['status'] == 'selected':
        cp = selection['checkpoint_identity']['path']
        if identity(cp) != selection['checkpoint_identity'] or ex.sha256(cp) != selection['checkpoint_sha256']:
            raise ValueError('selected merge changed during test evaluation')
        if ac.base.preflight(ac.load_config(args.config)) != data['metadata']['input_provenance']:
            raise ValueError('fixed data changed during test evaluation')
        for target in ac.base.SOURCES:
            report = control.read(root / 'results/node_neighbors_bias' / f'merge_selected__to__{target}.json')
            if (report['status'] != 'complete' or report['target'] != target
                    or report['provenance']['checkpoint'] != selection['checkpoint_identity']
                    or report['provenance']['pair_reference'] != data['metadata']['input_provenance'][target]['evaluation_pairs']):
                raise ValueError('selected-merge test report provenance mismatch')
            rows.append(dict(arm='merge_selected', target=target, checkpoint_step=report['checkpoint_step'],
                             **{k: report['metrics']['test'][k] for k in ('roc_auc', 'bce', 'average_precision')}))
    output = root / 'results'
    output.mkdir(exist_ok=True)
    with (output / 'matrix.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['arm', 'target', 'checkpoint_step', 'roc_auc', 'bce', 'average_precision'])
        writer.writeheader(); writer.writerows(rows)
    atomic_json(output / 'COMPLETE.json', dict(status='complete', outcome=selection['status'],
                source_validation_candidates=11, new_training_updates=0, evaluation_cells=len(rows),
                selected_alpha=selection['selected_alpha']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['plan', 'validate', 'eval', 'aggregate'])
    p.add_argument('--root', required=True)
    p.add_argument('--control-root', default='/dataMeR1/phil/gfm/mixture-scaling/state/async_seed2_control_s2')
    p.add_argument('--config', default='configs/nonzero_mini_transfer.yaml')
    p.add_argument('--device', type=int, choices=range(4), default=0)
    p.add_argument('--prodigy-root', default='/dataMeR1/phil/gfm/prodigy-walk-mini-pilot')
    args = p.parse_args()
    if args.phase == 'plan':
        _, _, endpoints = inputs(args)
        print(dict(alphas=ALPHAS, rule=RULE, endpoints=endpoints))
    elif args.phase == 'aggregate':
        aggregate(args)
    elif args.phase == 'eval':
        evaluate(args)
    else:
        torch.set_num_threads(4); torch.cuda.set_device(args.device); torch.backends.cuda.matmul.allow_tf32 = False
        validate(args, torch.device(f'cuda:{args.device}'))


if __name__ == '__main__':
    main()
