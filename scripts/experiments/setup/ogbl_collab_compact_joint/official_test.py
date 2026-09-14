"""User-authorized, explicitly test-tuned compact fusion; no backbone training."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch

import run as joint
from fusion import cutoff, hits, maximum, transform

ALPHAS = (0., .001, .0025, .005, .01, .025, .05, .1, .2, .3, .5, .75,
          1., 1.5, 2., 3., 5., 10., 30., 100.)
BASES = ('frozen2015', 'validation2018')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, default=Path('/dataMeR1/phil/data/ogb'))
    p.add_argument('--upstream', type=Path, default=Path('/dataMeR1/phil/gfm/ogbl_collab_aadc/official_b499c204/upstream'))
    p.add_argument('--device', choices=['cuda:0','cuda:1','cuda:2','cuda:3'], default='cuda:0')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    protocol = dict(name='official_test_fusion_v1', classification='test_tuned_development',
        test_scored=True, test_used_for_selection=True, training=False,
        seeds=[0, 1, 2], checkpoints='existing 2017-selected joint best.pt only',
        bases=BASES, alphas=ALPHAS, joint_only_control=True,
        selection='maximum mean 2019 Hits across3 seeds, then declared base/alpha order; report individual maxima separately',
        normalization='each scorer at its own 2019 50th negative; deliberately test-dependent',
        fusion='max(AA/AA_negative50, alpha*exp(clip(joint_logit-joint_negative50,-30,30)))',
        graph='train+2018 validation only; no2019 events; unique unweighted undirected GNN adjacency',
        aa_weights='upstream train weight lookup and decay .95 with reference2018; unit2018 validation weights',
        model_inputs='original symmetric14 with2015 AA calibration and2016 standardization; static supplied128-D features',
        total_inference_scalar_upper_bound=496448,
        scalar_count='496385 neural +44 original auxiliary +16 validation AA calibration +2 normalization +1 alpha',
        ensemble=False, target_test_hits=.7129,
        stop='one fixed grid after3 checkpoint inferences; no retries, additional training or automatic extension')
    if args.dry_run:
        import json
        print(json.dumps(protocol, indent=2))
        return
    torch.set_num_threads(8)
    args.out.mkdir(parents=True, exist_ok=False)
    protocol['revision'] = joint.revision()
    joint.save(args.out/'protocol.json', protocol)
    import wandb
    tracking = wandb.init(project='ogbl-collab-compact-joint', group='official_test_fusion_v1',
                          name=args.out.name, mode='offline', dir=str(args.out), config=protocol)
    started = time.monotonic()
    prepared = joint.read(args.runtime/'prepared.json')
    assert prepared['complete']
    for name in ('node_features.npy', 'standardization.npz'):
        assert joint.shared.sha256_file(args.runtime/name) == prepared['files'][name]
    frozen = joint.read(args.runtime/'assessment/selection_frozen.json')
    selections = [s for s in frozen['selections'] if s['arm'] == 'joint']
    assert sorted(s['seed'] for s in selections) == [0, 1, 2]
    for s in selections:
        assert s['revision'] == prepared['revision']
        assert joint.shared.sha256_file(args.runtime/f"joint_seed{s['seed']}/best.pt") == s['checkpoint_sha256']
    joint.save(args.out/'sources.json', dict(producer_revision=prepared['revision'], selections=selections,
         prepared_sha256=joint.shared.sha256_file(args.runtime/'prepared.json')))
    graph, split, evaluator, version = joint.shared.load_official_dataset(args.dataset_root)
    identity = joint.shared.audit_dataset(graph, split, version)
    assert identity['split_fingerprint'] == joint.gate.FINGERPRINT
    import subprocess
    assert subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip() == joint.gate.UPSTREAM
    source = args.upstream/'aa_dc.py'
    assert source.read_bytes() == subprocess.check_output(['git', '-C', str(args.upstream), 'show', 'HEAD:aa_dc.py'])
    aa = joint.module(source, 'official_aa')
    # Reproduce validation calibration rather than fitting AA on test positives.
    _, validation_meta, reference = joint.gate.make_year(aa, joint.shared, graph, split, 2018, None)
    panel, meta, _ = joint.gate.make_year(aa, joint.shared, graph, split, 2019,
         prepared['calibration'], symmetric_features=True, allow_test=True, reference_calibration=reference)
    edges, years, weights, _ = joint.gate.official_test_inputs(graph, split)
    panel['graph_edges'] = np.unique(np.sort(edges, axis=1), axis=0)
    assert np.max(years) == 2018
    np.testing.assert_array_equal(panel['pos'], split['test']['edge'])
    np.testing.assert_array_equal(panel['neg'], split['test']['edge_neg'])
    assert len(panel['neg']) == 100000
    np.savez_compressed(args.out/'year2019.npz', **panel)
    count = joint.COUNTS['joint'] + prepared['fitted_auxiliary_scalars'] + joint.numeric_count(reference) + 3
    assert count <= protocol['total_inference_scalar_upper_bound'] < 1_000_000
    joint.save(args.out/'panel.json', dict(metadata=meta, validation_metadata=validation_meta,
         identity=identity, reference_calibration=reference, total_inference_scalars=count,
         graph_events_fingerprint=joint.shared.fingerprint(edges, years, weights),
         graph_edges_fingerprint=joint.shared.fingerprint(panel['graph_edges']),
         upstream_revision=joint.gate.UPSTREAM,
         upstream_source_sha256=joint.shared.sha256_file(source),
         panel_sha256=joint.shared.sha256_file(args.out/'year2019.npz')))
    ms = np.load(args.runtime/'standardization.npz')
    x = torch.as_tensor(np.load(args.runtime/'node_features.npy'), device=args.device)
    np.testing.assert_array_equal(x.cpu().numpy(), graph['node_feat'])
    adj = joint.adjacency(panel['graph_edges'], len(x), args.device)
    tensors = joint.panel_tensors(panel, ms['mean'], ms['std'], args.device)
    arrays, records = {}, []
    for s in sorted(selections, key=lambda s: s['seed']):
        seed = s['seed']
        state = torch.load(args.runtime/f'joint_seed{seed}/best.pt', map_location=args.device, weights_only=False)
        assert state['step'] == s['selected_step'] and state['revision'] == s['revision']
        model = joint.Model('joint').to(args.device)
        model.load_state_dict(state['model'], strict=True)
        values = joint.evaluate(model, x, adj, tensors)
        evaluator.K = 50
        metric = evaluator.eval({'y_pred_pos': values['p'], 'y_pred_neg': values['n']})['hits@50']
        assert hits(**values) == metric
        for side in ('p','n'):
            e = panel['pos' if side=='p' else 'neg']
            assert (values[side][e[:,0] == e[:,1]] == -1e9).all()
            arrays[f'joint_seed{seed}_{side}'] = values[side]
        records.append(dict(seed=seed, selected_step=s['selected_step'], joint_hits=metric))
        del model
    np.savez_compressed(args.out/'scores.npz', **arrays)
    grids = []
    baseline = {}
    for base in BASES:
        b = {s: panel[('official_b' if base=='validation2018' else 'b')+s] for s in ('p','n')}
        baseline[base] = hits(**b)
        normalized = [transform({'bp':b['p'], 'bn':b['n']},
             {s:arrays[f'joint_seed{seed}_{s}'] for s in ('p','n')},
             cutoff(b['n']), cutoff(arrays[f'joint_seed{seed}_n'])) for seed in range(3)]
        for alpha in ALPHAS:
            rows = []
            for seed, z in enumerate(normalized):
                f = maximum(z, alpha)
                h = f['p'] > cutoff(f['n'])
                bh = b['p'] > cutoff(b['n'])
                official = evaluator.eval({'y_pred_pos':f['p'], 'y_pred_neg':f['n']})['hits@50']
                assert hits(**f) == official
                rows.append(dict(seed=seed, hits=official, recovered=int((h & ~bh).sum()),
                                 lost=int((bh & ~h).sum())))
            grids.append(dict(base=base, alpha=alpha, rows=rows,
                mean=float(np.mean([r['hits'] for r in rows])), sample_sd=float(np.std([r['hits'] for r in rows], ddof=1))))
    best = max(grids, key=lambda g: g['mean'])
    best_single = max((dict(base=g['base'], alpha=g['alpha'], **r) for g in grids for r in g['rows']), key=lambda r:r['hits'])
    result = dict(complete=True, classification='test_tuned_development', test_scored=True,
        test_used_for_selection=True, baseline=baseline, joint=records, grid=grids,
        best_common=best, best_single=best_single, target_test_hits=.7129,
        mean_target_exceeded=best['mean']>.7129, single_target_exceeded=best_single['hits']>.7129,
        panel_sha256=joint.shared.sha256_file(args.out/'year2019.npz'),
        score_sha256=joint.shared.sha256_file(args.out/'scores.npz'),
        total_inference_scalars=count, elapsed_seconds=time.monotonic()-started, wandb_directory=tracking.dir)
    joint.save(args.out/'results.json', result)
    tracking.log({'best_common_test_hits':best['mean'], 'best_single_test_hits':best_single['hits']})
    tracking.finish()
    import json
    print(json.dumps({k:v for k,v in result.items() if k != 'grid'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
