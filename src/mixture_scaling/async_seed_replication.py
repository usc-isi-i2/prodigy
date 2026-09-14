"""Locked-weight replication across training seeds with fixed evaluation data."""
import argparse
import copy
import csv
import json
import shutil
from pathlib import Path

import torch

from . import async_convergence as ac
from . import async_extension as ex
from .evaluation_artifacts import atomic_json

SEEDS = (1, 2)
WEIGHTS = {'kd_w010': .1, 'kd_w100': 1.}


def equal_trees(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, (list,tuple)):
        return isinstance(b, (list,tuple)) and len(a) == len(b) and all(equal_trees(x,y) for x,y in zip(a,b))
    if isinstance(a, dict):
        return isinstance(b,dict) and a.keys() == b.keys() and all(equal_trees(a[k],b[k]) for k in a)
    return a == b


def train_continuation(args):
    seedroot = Path(args.root) / f'seed{args.training_seed}'
    parent = seedroot / 'parent'
    summary = json.loads((parent / 'node_neighbors/lp/async_kd/summary.json').read_text())
    if summary['stop_reason'] != 'all_tasks_converged':
        atomic_json(seedroot / f'not_applicable_{args.run_id}.json',
                    dict(status='not_applicable', reason='parent did not reach all-task convergence', parent_summary=summary))
        return
    child = copy.copy(args)
    child.root = str(seedroot / 'continuations')
    child.parent_root = str(parent)
    child.arm = 'kd_extended'
    child.seed = args.training_seed
    child.data_seed = 0
    child.kd_weight = WEIGHTS[args.run_id]
    ex.train(child, torch.device(f'cuda:{args.device}'))


def prepare_eval(args):
    root = Path(args.root)
    aggregate = {'seeds':{}}
    shared_probes = None
    shared_receipts = None
    for seed in SEEDS:
        seedroot = root / f'seed{seed}'
        parent = seedroot / 'parent/node_neighbors/lp/async_kd'
        parent_summary = json.loads((parent / 'summary.json').read_text())
        if parent_summary['seed'] != seed or parent_summary['data_seed'] != 0 or parent_summary['probe_seed'] != 0:
            raise ValueError('parent seed roles differ from locked replication')
        if parent_summary['status'] != 'complete':
            raise ValueError('incomplete parent')
        probes = [torch.load(parent / f'probe_{s}.pt', map_location='cpu', weights_only=False) for s in ac.SOURCES]
        receipts = parent_summary['graph_split_receipts']
        if shared_probes is not None and (not equal_trees(shared_probes,probes) or shared_receipts != receipts):
            raise ValueError('training seeds changed fixed data/probe identities')
        shared_probes, shared_receipts = probes, receipts
        singletons, thresholds, models, unavailable = {}, [], [], []
        evalroot = seedroot / 'evaluation'
        def export(alias, arm, path, validation, counts, selection, additional=None, weight=None):
            dest = evalroot / 'node_neighbors/lp' / alias
            dest.mkdir(parents=True, exist_ok=False)
            shutil.copyfile(path, dest / 'best.pt')
            entry = dict(run_id=alias, arm=arm, original_checkpoint=str(path), selection=selection,
                         validation=validation, counts=counts, additional_step=additional, weight=weight,
                         start_fallback=additional==0)
            models.append(entry)
        for i, source in enumerate(ac.SOURCES):
            run = seedroot / 'singletons' / source
            summary = json.loads((run / 'summary.json').read_text())
            history = ex.read_history(run / 'history.jsonl')
            if summary['seed'] != seed or summary['data_seed'] != 0 or summary['status'] != 'complete':
                raise ValueError('singleton seed/status mismatch')
            if summary['graph_split_receipt'] != receipts[source]:
                raise ValueError('singleton and joint graph identities differ')
            probe = torch.load(run / 'training_probe.pt', map_location='cpu', weights_only=False)
            if not equal_trees(probe, probes[i]):
                raise ValueError('singleton and joint fixed probes differ')
            row = next(r for r in history if r['step'] == summary['selected_step'])
            if row['validation'] != summary['selected_validation']:
                raise ValueError('singleton selected row differs from summary')
            singletons[source] = dict(summary=summary, history=history)
            thresholds.append(row['validation']['auc'])
            export('singleton_' + ('ukraine' if i==0 else 'facebook'), 'singleton', row['checkpoint'],
                   {source:row['validation']}, {source:row['counts']}, 'seed-specific minimum source-validation BCE')
        histories, summaries = {}, {}
        matched = False
        if parent_summary['stop_reason'] == 'all_tasks_converged':
            endpoints = []
            for arm, weight in WEIGHTS.items():
                run = seedroot / 'continuations/node_neighbors/lp' / arm
                rows = ex.read_history(run / 'history.jsonl')
                summary = json.loads((run / 'summary.json').read_text())
                if (summary['status'] != 'complete' or summary['seed'] != seed or summary['data_seed'] != 0
                        or summary['probe_seed'] != 0 or summary['protocol']['kd_weight'] != weight
                        or summary['budget_steps'] != args.additional_steps):
                    raise ValueError('continuation protocol mismatch')
                if summaries:
                    other = next(iter(summaries.values()))
                    for key in ('start_sha256','start_counts','teacher_sha256','probe_sha256','surviving_counts'):
                        if summary[key] != other[key]:
                            raise ValueError(f'paired continuations differ in {key}')
                    for key in ('learning_rate','weight_decay','validation_interval','schedule','kd_temperature','optimizer'):
                        if summary['protocol'][key] != other['protocol'][key]:
                            raise ValueError(f'paired continuation protocol differs in {key}')
                summaries[arm], histories[arm] = summary, rows
                fixed = rows[-1]
                if fixed['additional_step'] != args.additional_steps:
                    raise ValueError('continuation fixed endpoint missing')
                endpoints.append(torch.load(fixed['checkpoint'], map_location='cpu', weights_only=False))
                export(arm+'_fixed',arm,fixed['checkpoint'],fixed['validation'],fixed['counts'],
                       'fixed added-update horizon', fixed['additional_step'],weight)
                selected = ex.select_preserving(rows,thresholds[1],summary['start_counts'][0]['supervised_updates'],
                                                args.additional_steps//2,args.selection_grid)
                if selected is None:
                    unavailable.append(dict(arm=arm,reason='no checkpoint meets seed-specific Facebook validation floor'))
                else:
                    export(arm+'_selected',arm,selected['checkpoint'],selected['validation'],selected['counts'],
                           'max Ukraine validation AUC subject to seed-specific Facebook singleton AUC; common exposure grid',
                           selected['additional_step'],weight)
            if not equal_trees(endpoints[0]['samplers'],endpoints[1]['samplers']):
                raise ValueError('paired continuation samplers differ')
            matched = True
        else:
            unavailable.extend(dict(arm=a,reason='parent did not reach all-task convergence') for a in WEIGHTS)
        manifest = dict(seed=seed,models=models,unavailable=unavailable,locked_weight=.1,
                        singleton_source_auc=dict(zip(ac.SOURCES,thresholds)),selection_grid=args.selection_grid,
                        selection_data='source validation only; no weight selection; frozen before downstream evaluation',
                        audit=dict(fixed_data_and_probes=True,paired_continuation_samplers_match=matched,
                                   parent_stop_reason=parent_summary['stop_reason']))
        atomic_json(evalroot / 'evaluation_manifest.json',manifest)
        aggregate['seeds'][str(seed)] = dict(parent_summary=parent_summary,singletons=singletons,
                                            history=histories,summary=summaries,manifest=manifest)
    # Also compare the fixed probe tensors to seed0, not serialized-file hashes.
    reference = Path(args.development_parent) / 'node_neighbors/lp/async_kd'
    reference_summary = json.loads((reference / 'summary.json').read_text())
    if shared_receipts != reference_summary['graph_split_receipts']:
        raise ValueError('replication graph/split identities differ from development seed0')
    original_probes = [torch.load(reference / f'probe_{s}.pt',map_location='cpu',weights_only=False) for s in ac.SOURCES]
    if not equal_trees(shared_probes,original_probes):
        raise ValueError('replication probes differ from development seed0')
    atomic_json(root / 'analysis_data.json',aggregate)
    print(json.dumps({s:d['manifest'] for s,d in aggregate['seeds'].items()},indent=2))


def aggregate_results(args):
    root=Path(args.root)
    rows=[]
    unavailable=[]
    for seed in SEEDS:
        evalroot=root / f'seed{seed}/evaluation'
        manifest=json.loads((evalroot / 'evaluation_manifest.json').read_text())
        unavailable.extend(dict(seed=seed,**r) for r in manifest['unavailable'])
        for model in manifest['models']:
            for target in ac.base.SOURCES:
                report=json.loads((evalroot / 'results/node_neighbors_bias'/f'{model["run_id"]}__to__{target}.json').read_text())
                rows.append(dict(seed=seed,arm=model['run_id'],target=target,checkpoint_step=report['checkpoint_step'],
                                 **{k:report['metrics']['test'][k] for k in ('roc_auc','bce','average_precision')}))
    output=root / 'results'
    output.mkdir(exist_ok=True)
    with (output / 'matrix.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    atomic_json(output / 'COMPLETE.json',dict(status='complete',training_seeds=list(SEEDS),data_seed=0,
                                             evaluation_cells=len(rows),unavailable=unavailable))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['plan','continue','prepare_eval','eval','aggregate'])
    p.add_argument('--root',required=True)
    p.add_argument('--training-seed',type=int,choices=SEEDS,default=1)
    p.add_argument('--run-id',choices=WEIGHTS,default='kd_w010')
    p.add_argument('--device',type=int,choices=range(4),default=0)
    p.add_argument('--config',default='configs/nonzero_mini_transfer.yaml')
    p.add_argument('--additional-steps',type=int,default=60000)
    p.add_argument('--selection-grid',type=int,default=2000)
    p.add_argument('--validation-interval',type=int,default=2000)
    p.add_argument('--log-interval',type=int,default=100)
    p.add_argument('--development-parent',default='/dataMeR1/phil/gfm/mixture-scaling/state/async_convergence_s0')
    p.add_argument('--wandb-mode',default='offline')
    p.add_argument('--wandb-project',default='nonzero-mini-transfer')
    p.add_argument('--wandb-group',default='async-seed-replication')
    p.add_argument('--prodigy-root',default='/dataMeR1/phil/gfm/prodigy-walk-mini-pilot')
    args=p.parse_args()
    if min(args.additional_steps,args.selection_grid,args.validation_interval,args.log_interval)<=0:
        p.error('positive steps and intervals required')
    if args.additional_steps % (2*args.selection_grid) or args.additional_steps % args.validation_interval:
        p.error('budget must divide exposure and validation grids')
    if args.phase=='plan':
        print(json.dumps(dict(training_seeds=SEEDS,data_seed=0,probe_seed=0,locked_weight=.1,
                              additional_updates=args.additional_steps,inputs=ac.base.preflight(ac.load_config(args.config))),indent=2))
    elif args.phase=='prepare_eval':prepare_eval(args)
    elif args.phase=='aggregate':aggregate_results(args)
    else:
        torch.set_num_threads(4);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False
        if args.phase=='continue':train_continuation(args)
        else:
            evalroot=Path(args.root)/f'seed{args.training_seed}/evaluation'
            args.root=str(evalroot);args.seed=0;args.views=('node_neighbors',);args.targets=ac.base.SOURCES
            args.model_rows=[(m['run_id'],m['run_id']) for m in json.loads((evalroot/'evaluation_manifest.json').read_text())['models']]
            ac.base.evaluate(args,torch.device(f'cuda:{args.device}'))


if __name__=='__main__':main()
