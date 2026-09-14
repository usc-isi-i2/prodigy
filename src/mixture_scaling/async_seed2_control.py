"""One Ukraine-only diagnostic branch from the existing seed-2 joint parent."""
import argparse
import copy
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from . import async_convergence as ac
from . import async_extension as ex
from .evaluation_artifacts import atomic_json
from .mini_transfer import identity

REUSED = ('kd_w010_fixed', 'kd_w010_selected', 'singleton_ukraine', 'singleton_facebook')
MATCHES = (('ukraine_exposure_fixed', 30000, 'kd_w010_fixed', 'Ukraine exposure'),
           ('ukraine_updates_fixed', 60000, 'kd_w010_fixed', 'optimizer updates'),
           ('ukraine_exposure_selected', 28000, 'kd_w010_selected', 'Ukraine exposure'),
           ('ukraine_updates_selected', 56000, 'kd_w010_selected', 'optimizer updates'))


def read(path):
    return json.loads(Path(path).read_text())


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return isinstance(b, (tuple, list)) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def checkpoint(row):
    return torch.load(row['checkpoint'], map_location='cpu', weights_only=False)


def reference(args):
    ref = Path(args.reference_root)
    manifest = read(ref / 'evaluation/evaluation_manifest.json')
    parent = read(ref / 'parent/node_neighbors/lp/async_kd/summary.json')
    if (manifest['seed'] != 2 or manifest['locked_weight'] != .1 or parent['seed'] != 2
            or parent['data_seed'] != 0 or parent['probe_seed'] != 0
            or parent['stop_reason'] != 'all_tasks_converged'):
        raise ValueError('expected the completed seed-2 locked-weight parent')
    models = {m['run_id']: m for m in manifest['models']}
    if models['kd_w010_fixed']['additional_step'] != 60000 or models['kd_w010_selected']['additional_step'] != 56000:
        raise ValueError('frozen seed-2 reference checkpoint changed')
    fb = read(ref / 'singletons/facebook_page_reference/summary.json')
    floor = manifest['singleton_source_auc']['facebook_page_reference']
    if floor != fb['selected_validation']['auc'] or floor != .9531494824886322:
        raise ValueError('seed-2 Facebook singleton floor changed')
    kd = ref / 'continuations/node_neighbors/lp/kd_w010'
    summary, rows = read(kd / 'summary.json'), ex.read_history(kd / 'history.jsonl')
    if (summary['status'] != 'complete' or summary['seed'] != 2 or summary['data_seed'] != 0
            or summary['probe_seed'] != 0 or summary['budget_steps'] != 60000
            or summary['protocol']['kd_weight'] != .1):
        raise ValueError('reused KD run differs from locked seed-2 recipe')
    for alias in ('kd_w010_fixed', 'kd_w010_selected'):
        entry = models[alias]
        row = next(r for r in rows if r['additional_step'] == entry['additional_step'])
        if (row['validation'] != entry['validation'] or row['counts'] != entry['counts']
                or ex.sha256(row['checkpoint']) != ex.sha256(entry['original_checkpoint'])):
            raise ValueError('KD history differs from its frozen checkpoint selection')
    return ref, parent, manifest, models, summary, rows


def audit_control(summary, rows, kd_summary, kd_rows):
    if (summary['status'] != 'complete' or summary['seed'] != 2 or summary['data_seed'] != 0
            or summary['probe_seed'] != 0 or summary['budget_steps'] != 60000):
        raise ValueError('control seed/data/budget mismatch')
    for key in ('start_sha256', 'start_counts', 'probe_sha256', 'graph_split_receipts'):
        if summary[key] != kd_summary[key]:
            raise ValueError(f'control and KD differ in {key}')
    for key in ('learning_rate', 'weight_decay', 'optimizer', 'validation_interval'):
        if summary['protocol'][key] != kd_summary['protocol'][key]:
            raise ValueError(f'control protocol differs in {key}')
    if summary['protocol']['schedule'] != 'ukraine_only' or summary['protocol']['kd_weight'] != 0:
        raise ValueError('control objective is not Ukraine only')
    if summary['teacher_forward_batches'] or any(summary['physical_counts'][1].values()):
        raise ValueError('control performed Facebook training or teacher forwards')
    if summary['physical_counts'][0]['supervised_updates'] != 60000 or summary['physical_counts'][0]['kd_updates']:
        raise ValueError('control did not perform exactly 60k Ukraine BCE updates')
    by_step = {r['additional_step']: r for r in rows}
    kd_by_step = {r['additional_step']: r for r in kd_rows}
    starts = [checkpoint(r[0]) for r in (by_step, kd_by_step)]
    for key in ('model', 'optimizer', 'samplers', 'rng'):
        if not equal(starts[0][key], starts[1][key]):
            raise ValueError(f'initial full state differs in {key}')
    for row in rows:
        added = row['additional_step']
        if row['counts'][1] != summary['start_counts'][1]:
            raise ValueError('Facebook exposure changed in Ukraine-only history')
        if row['counts'][0]['supervised_updates'] != summary['start_counts'][0]['supervised_updates'] + added:
            raise ValueError('Ukraine-only exposure does not equal added updates')
    endpoint = checkpoint(by_step[60000])
    if not equal(endpoint['samplers'][1], starts[0]['samplers'][1]):
        raise ValueError('Facebook sampler advanced')
    for kd_step in (56000, 60000):
        control, kd = by_step[kd_step // 2], kd_by_step[kd_step]
        if control['counts'][0] != kd['counts'][0] or not equal(checkpoint(control)['samplers'][0], checkpoint(kd)['samplers'][0]):
            raise ValueError(f'Ukraine exposure/sampler mismatch at KD +{kd_step}')
    return dict(start_model_adam_sampler_rng_identical=True, zero_added_facebook_training=True,
                facebook_sampler_unchanged=True, ukraine_exposure_and_sampler_match_at=[56000, 60000])


def prepare_eval(args):
    root = Path(args.root)
    ref, parent, old_manifest, old_models, kd_summary, kd_rows = reference(args)
    run = root / 'node_neighbors/lp/ukraine_only'
    summary, rows = read(run / 'summary.json'), ex.read_history(run / 'history.jsonl')
    audit = audit_control(summary, rows, kd_summary, kd_rows)
    by_step = {r['additional_step']: r for r in rows}
    models, unavailable = [], []
    for alias in REUSED:
        entry = copy.deepcopy(old_models[alias])
        cp = ref / 'evaluation/node_neighbors/lp' / alias / 'best.pt'
        if ex.sha256(cp) != ex.sha256(entry['original_checkpoint']):
            raise ValueError('reference export differs from frozen selected checkpoint')
        entry.update(checkpoint_identity=identity(cp), checkpoint_sha256=ex.sha256(cp),
                     provenance=dict(kind='reused', evaluation_root=str(ref / 'evaluation')))
        models.append(entry)

    def export(alias, row, selection):
        cp = root / 'node_neighbors/lp' / alias / 'best.pt'
        cp.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(row['checkpoint'], cp)
        models.append(dict(run_id=alias, arm='ukraine_only', additional_step=row['additional_step'],
                           counts=row['counts'], validation=row['validation'], selection=selection,
                           original_checkpoint=row['checkpoint'], start_fallback=row['additional_step'] == 0,
                           checkpoint_identity=identity(cp), checkpoint_sha256=ex.sha256(cp),
                           provenance=dict(kind='new', evaluation_root=str(root))))
    for alias, step, comparator, basis in MATCHES:
        export(alias, by_step[step], f'match {basis} to frozen {comparator}; optimizer updates are not compute')
    selected = ex.select_preserving(rows, old_manifest['singleton_source_auc'][ac.SOURCES[1]],
                                    summary['start_counts'][0]['supervised_updates'], 30000, 2000)
    if selected is None:
        unavailable.append(dict(run_id='ukraine_only_selected', reason='no checkpoint meets Facebook singleton validation floor'))
    else:
        export('ukraine_only_selected', selected,
               'max Ukraine validation AUC subject to original seed2 Facebook floor; <=30k added Ukraine updates, 2k grid')
    manifest = dict(models=models, unavailable=unavailable, audit=audit, replication_seed=2,
                    singleton_source_auc=old_manifest['singleton_source_auc'],
                    selection_data='source validation only; existing KD selections frozen before this diagnostic',
                    input_provenance=ac.base.preflight(ac.load_config(args.config)))
    atomic_json(root / 'evaluation_manifest.json', manifest)
    atomic_json(root / 'analysis_data.json', dict(replication_seed=2,
                history=dict(ukraine_only=rows, kd_w010=kd_rows),
                summary=dict(ukraine_only=summary, kd_w010=kd_summary), parent_summary=parent, manifest=manifest))
    print(json.dumps(manifest, indent=2))


def aggregate(args):
    root = Path(args.root)
    manifest = read(root / 'evaluation_manifest.json')
    if manifest['input_provenance'] != ac.base.preflight(ac.load_config(args.config)):
        raise ValueError('data inputs changed after selection freeze')
    rows, pair_provenance, cells = [], {}, {'new': 0, 'reused': 0}
    for model in manifest['models']:
        if (identity(model['checkpoint_identity']['path']) != model['checkpoint_identity']
                or ex.sha256(model['checkpoint_identity']['path']) != model['checkpoint_sha256']):
            raise ValueError('exported checkpoint changed after freeze')
        for target in ac.base.SOURCES:
            report = read(Path(model['provenance']['evaluation_root']) / 'results/node_neighbors_bias' /
                          f'{model["run_id"]}__to__{target}.json')
            provenance = report['provenance']
            if (report['status'] != 'complete' or report['target'] != target
                    or provenance['checkpoint'] != model['checkpoint_identity']
                    or provenance['pair_reference'] != manifest['input_provenance'][target]['evaluation_pairs']):
                raise ValueError('test report identity mismatch')
            shared = {k: provenance[k] for k in ('pair_reference', 'split', 'decoder', 'bias_policy')}
            if target in pair_provenance and shared != pair_provenance[target]:
                raise ValueError('new/reused evaluation pair or split differs')
            pair_provenance[target] = shared
            cells[model['provenance']['kind']] += 1
            rows.append(dict(arm=model['run_id'], target=target, checkpoint_step=report['checkpoint_step'],
                             **{k: report['metrics']['test'][k] for k in ('roc_auc', 'bce', 'average_precision')}))
    output = root / 'results'
    output.mkdir(exist_ok=True)
    with (output / 'matrix.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    atomic_json(output / 'COMPLETE.json', dict(status='complete', training_seed=2, data_seed=0,
                trained_branches=1, evaluation_cells=cells, unavailable=manifest['unavailable']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['plan', 'train', 'prepare_eval', 'eval', 'aggregate'])
    p.add_argument('--root', required=True)
    p.add_argument('--reference-root', default='/dataMeR1/phil/gfm/mixture-scaling/state/async_seed_replication_s12/seed2')
    p.add_argument('--device', type=int, choices=range(4), default=0)
    p.add_argument('--config', default='configs/nonzero_mini_transfer.yaml')
    p.add_argument('--prodigy-root', default='/dataMeR1/phil/gfm/prodigy-walk-mini-pilot')
    p.add_argument('--wandb-mode', default='offline')
    p.add_argument('--wandb-project', default='nonzero-mini-transfer')
    p.add_argument('--wandb-group', default='async-seed2-control')
    args = p.parse_args()
    if args.phase == 'plan':
        _, parent, _, _, _, _ = reference(args)
        print(json.dumps(dict(training_seed=2, data_seed=0, additional_updates=60000, start_counts=parent['surviving_counts'],
                              matches=MATCHES, inputs=ac.base.preflight(ac.load_config(args.config))), indent=2))
    elif args.phase == 'prepare_eval':
        prepare_eval(args)
    elif args.phase == 'aggregate':
        aggregate(args)
    else:
        torch.set_num_threads(4); torch.cuda.set_device(args.device); torch.backends.cuda.matmul.allow_tf32 = False
        device = torch.device(f'cuda:{args.device}')
        if args.phase == 'train':
            reference(args)
            args.parent_root = str(Path(args.reference_root) / 'parent')
            args.arm = args.run_id = 'ukraine_only'; args.seed = 2; args.data_seed = 0
            args.additional_steps = 60000; args.validation_interval = 2000; args.log_interval = 100
            args.kd_weight = 1.
            ex.train(args, device)
        else:
            manifest = read(Path(args.root) / 'evaluation_manifest.json')
            args.seed = 0; args.views = ('node_neighbors',); args.targets = ac.base.SOURCES
            args.model_rows = [(m['run_id'], m['run_id']) for m in manifest['models'] if m['provenance']['kind'] == 'new']
            ac.base.evaluate(args, device)


if __name__ == '__main__':
    main()
