"""Lower-KD-weight comparison with source-only weight/checkpoint selection."""
import argparse
import json
import shutil
from pathlib import Path

import torch

from . import async_extension as ex
from . import async_convergence as ac
from .evaluation_artifacts import atomic_json

WEIGHTS = {'kd_w010': .1, 'kd_w030': .3, 'kd_w050': .5, 'kd_w100': 1.}


def prepare_eval(args):
    root = Path(args.root)
    history, summaries = {}, {}
    for arm, weight in WEIGHTS.items():
        run = (Path(args.reference_root) / 'node_neighbors/lp/kd_extended' if weight == 1.
               else root / 'node_neighbors/lp' / arm)
        history[arm] = ex.read_history(run / 'history.jsonl')
        summaries[arm] = json.loads((run / 'summary.json').read_text())
        if summaries[arm]['status'] != 'complete' or summaries[arm]['protocol']['kd_weight'] != weight:
            raise ValueError(f'incomplete or incorrect weight: {arm}')
    reference = summaries['kd_w100']
    for arm, summary in summaries.items():
        for key in ('seed', 'parent_root', 'start_sha256', 'start_counts', 'probe_sha256',
                    'graph_split_receipts', 'budget_steps', 'teacher_sha256', 'surviving_counts'):
            if summary[key] != reference[key]:
                raise ValueError(f'{arm} differs in controlled field {key}')
        for key in ('schedule','learning_rate','weight_decay','optimizer','kd_temperature','validation_interval'):
            if summary['protocol'][key] != reference['protocol'][key]:
                raise ValueError(f'{arm} differs in protocol {key}')
        if summary['budget_steps'] != args.additional_steps:
            raise ValueError('requested and completed budgets differ')
    reference_end = torch.load(history['kd_w100'][-1]['checkpoint'], map_location='cpu', weights_only=False)
    for arm, rows in history.items():
        endpoint = torch.load(rows[-1]['checkpoint'], map_location='cpu', weights_only=False)
        for left, right in zip(reference_end['samplers'], endpoint['samplers']):
            if left['offset'] != right['offset'] or any(not torch.equal(left[k],right[k]) for k in ('order','generator','order_generator')):
                raise ValueError(f'{arm} sampling differs at matched endpoint')
    references, thresholds = ex.reference_metrics(args.parent_root)
    models, unavailable, selected = [], [], []
    def export(arm, row, role):
        alias = arm + '_' + role
        dest = root / 'node_neighbors/lp' / alias
        dest.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(row['checkpoint'], dest / 'best.pt')
        entry = dict(run_id=alias, arm=arm, weight=WEIGHTS[arm], selection=role,
                     original_checkpoint=row['checkpoint'], additional_step=row['additional_step'],
                     counts=row['counts'], validation=row['validation'], start_fallback=row['additional_step']==0)
        models.append(entry)
        return entry
    for arm, rows in history.items():
        if rows[-1]['additional_step'] != args.additional_steps:
            raise ValueError(f'{arm} missing fixed endpoint')
        export(arm, rows[-1], 'fixed')
        row = ex.select_preserving(rows, thresholds[1], reference['start_counts'][0]['supervised_updates'],
                                   args.additional_steps//2, args.selection_grid)
        if row is None:
            unavailable.append(dict(arm=arm, reason='no Facebook-preserving source-validation candidate'))
        else:
            selected.append(export(arm, row, 'selected'))
    # Across weights: maximize Ukraine source validation, earliest checkpoint,
    # then larger weight for exact ties. No downstream scores are read here.
    winner = max(selected, key=lambda r:(r['validation'][0]['auc'], -r['additional_step'], r['weight'])) if selected else None
    manifest = dict(models=models, unavailable=unavailable, chosen_run_id=winner['run_id'] if winner else None,
                    singleton_source_auc=dict(zip(ac.SOURCES,thresholds)), selection_grid_ukraine_updates=args.selection_grid,
                    selection_rule='maximize Ukraine validation AUC subject to Facebook singleton validation AUC; common exposure grid; earliest step then largest weight for ties',
                    selection_data='source validation only; weight and checkpoints frozen before downstream evaluation',
                    replay_check=dict(same_start_teacher_inputs_and_budget=True, both_source_final_sampler_states_identical=True,
                                      weight_one_reused_from=args.reference_root, weight_one_parent_replay=reference['replay_checks']))
    atomic_json(root / 'evaluation_manifest.json', manifest)
    atomic_json(root / 'analysis_data.json', dict(history=history,summary=summaries,manifest=manifest,references=references))
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['plan','prepare_eval','eval','aggregate'])
    parser.add_argument('--root', required=True)
    parser.add_argument('--parent-root', required=True)
    parser.add_argument('--reference-root', required=True)
    parser.add_argument('--config', default='configs/nonzero_mini_transfer.yaml')
    parser.add_argument('--additional-steps', type=int, default=60000)
    parser.add_argument('--selection-grid', type=int, default=2000)
    parser.add_argument('--device', type=int, choices=range(4), default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prodigy-root', default='/dataMeR1/phil/gfm/prodigy-walk-mini-pilot')
    args = parser.parse_args()
    if args.additional_steps <= 0 or args.selection_grid <= 0 or args.additional_steps % (2*args.selection_grid):
        parser.error('positive budget divisible by twice the source-exposure grid required')
    if args.phase == 'plan':
        _, _, path, _ = ex.load_start(args.parent_root)
        reference = json.loads((Path(args.reference_root)/'node_neighbors/lp/kd_extended/summary.json').read_text())
        if reference['status'] != 'complete' or reference['budget_steps'] != args.additional_steps:
            raise ValueError('completed equal-horizon weight-one reference required')
        print(json.dumps(dict(weights=WEIGHTS, start=str(path), reference_root=args.reference_root,
                              additional_optimizer_steps=args.additional_steps, inputs=ac.base.preflight(ac.load_config(args.config))),indent=2))
    elif args.phase == 'prepare_eval':
        prepare_eval(args)
    elif args.phase == 'aggregate':
        ac.aggregate(args)
    else:
        torch.set_num_threads(4)
        torch.cuda.set_device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        args.views = ('node_neighbors',)
        args.targets = ac.base.SOURCES
        args.model_rows = [(m['run_id'],m['run_id']) for m in json.loads((Path(args.root)/'evaluation_manifest.json').read_text())['models']]
        ac.base.evaluate(args, torch.device(f'cuda:{args.device}'))


if __name__ == '__main__':
    main()
