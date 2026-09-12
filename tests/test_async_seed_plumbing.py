import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mixture_scaling import async_convergence as ac
from mixture_scaling import async_extension as ex
from mixture_scaling import async_seed_replication as replication
from tests.test_async_extension import TinyModel, graph, make_parent, tracking


class SeedPlumbingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def assert_parts_equal(self, left, right):
        self.assertEqual(len(left), len(right))
        for first, second in zip(left, right):
            self.assertTrue(torch.equal(first[0], second[0]))
            self.assertTrue(torch.equal(first[1], second[1]))

    def test_optional_seed_keeps_zero_and_preserves_legacy_defaults(self):
        self.assertEqual(ac.optional_seed(SimpleNamespace(), 'data_seed', 7), 7)
        self.assertEqual(ac.optional_seed(SimpleNamespace(data_seed=None), 'data_seed', 7), 7)
        self.assertEqual(ac.optional_seed(SimpleNamespace(data_seed=0), 'data_seed', 7), 0)

    def test_parent_varies_training_seed_with_identical_data_probes_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            outputs, initial_states, final_states, validation_parts = {}, {}, {}, {}
            original_validation = ac.diagnostic.validation_pairs
            original_probe = ac.fixed_probe
            for seed in (1, 2):
                loaded, probe_seeds = [], []
                validation_parts[seed] = []
                root = Path(temporary) / f'seed{seed}'
                args = SimpleNamespace(root=str(root), arm='extended_bce', config='unused', seed=seed,
                                       data_seed=0, probe_seed=0, learning_rate=.0005, max_steps=2,
                                       validation_interval=2, patience=3, min_delta=.0001,
                                       minimum_steps=2500, log_interval=2)

                def load(source, config, data_seed, device):
                    loaded.append(data_seed)
                    return graph(source)

                def validation(g):
                    parts = original_validation(g)
                    validation_parts[seed].append(ac.cpu_clone(parts))
                    return parts

                def probe(g, probe_seed):
                    probe_seeds.append(probe_seed)
                    return original_probe(g, probe_seed)

                with patch.object(ac.base, 'BiasMLP', TinyModel), \
                     patch.object(ac, 'load_config', return_value={}), \
                     patch.object(ac.base, 'preflight'), \
                     patch.object(ac.base, 'load_graph', side_effect=load), \
                     patch.object(ac.diagnostic, 'validation_pairs', side_effect=validation), \
                     patch.object(ac, 'fixed_probe', side_effect=probe), \
                     patch.object(ac, 'tracked_run', side_effect=tracking), \
                     contextlib.redirect_stdout(io.StringIO()):
                    ac.train(args, torch.device('cpu'))
                self.assertEqual(loaded, [0, 0])
                self.assertEqual(probe_seeds, [813719, 814728])
                run = root / 'node_neighbors/lp/extended_bce'
                outputs[seed] = run
                initial = torch.load(run / 'checkpoints/physical_000000_logical_000000.pt', weights_only=False)
                final = torch.load(run / 'endpoint.pt', weights_only=False)
                initial_states[seed], final_states[seed] = initial, final
                self.assertEqual(initial['metadata']['training_seed'], seed)
                self.assertEqual(initial['metadata']['data_seed'], 0)
                self.assertEqual(initial['metadata']['probe_seed'], 0)
                torch.manual_seed(seed)
                expected_model = TinyModel()
                for name, tensor in expected_model.state_dict().items():
                    self.assertTrue(torch.equal(tensor, initial['model'][name]))
                for sampler in initial['samplers']:
                    for field, offset in (('generator', 49979687), ('order_generator', 7919)):
                        expected_state = torch.Generator().manual_seed(seed + offset).get_state()
                        self.assertTrue(torch.equal(sampler[field], expected_state))
            self.assertTrue(any(not torch.equal(value, initial_states[2]['model'][key])
                                for key, value in initial_states[1]['model'].items()))
            self.assertFalse(torch.equal(final_states[1]['samplers'][0]['order'],
                                         final_states[2]['samplers'][0]['order']))
            for index, source in enumerate(ac.SOURCES):
                first = torch.load(outputs[1] / f'probe_{source}.pt', weights_only=False)
                second = torch.load(outputs[2] / f'probe_{source}.pt', weights_only=False)
                self.assert_parts_equal(first, second)
                self.assert_parts_equal(validation_parts[1][index], validation_parts[2][index])

    def test_endpoint_supplies_facebook_teacher_absent_from_raw_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            make_parent(parent)
            run = parent / 'node_neighbors/lp/async_kd'
            selected_path = run / 'selected.pt'
            raw = torch.load(selected_path, weights_only=False)
            raw['runtime']['converged'] = [True, False]
            raw['runtime']['teacher_paths'][1] = None
            ac.save(selected_path, raw)
            endpoint_path = run / 'endpoint.pt'
            endpoint = torch.load(endpoint_path, weights_only=False)
            endpoint['runtime']['teacher_paths'][1] = str(selected_path)
            ac.save(endpoint_path, endpoint)
            _, _, loaded_path, loaded = ex.load_start(parent)
            self.assertEqual(loaded_path, endpoint_path)
            self.assertEqual(loaded['runtime']['converged'], [True, True])
            self.assertEqual(loaded['runtime']['teacher_paths'][1], str(selected_path))
            summary_path = run / 'summary.json'
            summary = json.loads(summary_path.read_text())
            summary['stop_reason'] = 'safety_cap'
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'all-converged parent'):
                ex.load_start(parent)

    def test_replay_requires_matching_objective_and_teacher_but_remains_exact(self):
        for case in ('opposite_objectives', 'different_teacher', 'matching_but_corrupted'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                parent = Path(temporary) / 'parent'
                output = Path(temporary) / 'extension'
                make_parent(parent, seed=1)
                prior_path = parent / 'node_neighbors/lp/async_kd/discarded_tail.pt'
                prior = torch.load(prior_path, weights_only=False)
                key = next(iter(prior['model']))
                prior['model'][key] = prior['model'][key] + 1
                if case == 'opposite_objectives':
                    prior['runtime']['converged'] = [True, False]
                elif case == 'different_teacher':
                    prior['runtime']['teacher_paths'][1] = 'different_teacher.pt'
                ac.save(prior_path, prior)
                loaded = []

                def load(source, config, data_seed, device):
                    loaded.append(data_seed)
                    return graph(source)

                args = SimpleNamespace(root=str(output), parent_root=str(parent), arm='kd_extended',
                                       config='unused', seed=1, data_seed=None, additional_steps=4,
                                       validation_interval=2, log_interval=2, selection_grid=2)
                with patch.object(ac.base, 'BiasMLP', TinyModel), \
                     patch.object(ac, 'load_config', return_value={}), \
                     patch.object(ac.base, 'preflight'), \
                     patch.object(ac.base, 'load_graph', side_effect=load), \
                     patch.object(ac, 'tracked_run', side_effect=tracking), \
                     contextlib.redirect_stdout(io.StringIO()):
                    if case == 'matching_but_corrupted':
                        with self.assertRaisesRegex(ValueError, 'prior KD replay failed'):
                            ex.train(args, torch.device('cpu'))
                    else:
                        ex.train(args, torch.device('cpu'))
                self.assertEqual(loaded, [0, 0])
                if case != 'matching_but_corrupted':
                    summary = json.loads((output / 'node_neighbors/lp/kd_extended/summary.json').read_text())
                    self.assertEqual(summary['training_seed'], 1)
                    self.assertEqual(summary['data_seed'], 0)
                    self.assertEqual(summary['probe_seed'], 0)
                    self.assertEqual(summary['replay_checks'], [])
                    self.assertEqual(len(summary['replay_not_applicable']), 1)

    def test_prepare_eval_uses_each_seed_floor_and_reports_capped_parent(self):
        for capped_second in (False, True):
            with self.subTest(capped_second=capped_second), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / 'replication'
                development = Path(temporary) / 'development'
                make_parent(development)
                for seed in (1, 2):
                    seedroot = root / f'seed{seed}'
                    parent = seedroot / 'parent'
                    initial = make_parent(parent, seed=seed)
                    parent_run = parent / 'node_neighbors/lp/async_kd'
                    if seed == 2 and capped_second:
                        summary_path = parent_run / 'summary.json'
                        parent_summary = json.loads(summary_path.read_text())
                        parent_summary['stop_reason'] = 'safety_cap'
                        summary_path.write_text(json.dumps(parent_summary))
                    for index, source in enumerate(ac.SOURCES):
                        run = seedroot / 'singletons' / source
                        run.mkdir(parents=True)
                        probe = torch.load(parent_run / f'probe_{source}.pt', weights_only=False)
                        ac.save(run / 'training_probe.pt', probe)
                        floor = .9 if seed == 1 else .7
                        validation = dict(auc=floor if index else .8, bce=.2)
                        counts = copy.deepcopy(initial['runtime']['counts'][index])
                        row = dict(step=4, checkpoint=str(parent_run / 'endpoint.pt'),
                                   validation=validation, counts=counts)
                        (run / 'history.jsonl').write_text(json.dumps(row) + '\n')
                        summary = dict(seed=seed, data_seed=0, status='complete', selected_step=4,
                                       selected_validation=validation, graph_split_receipt=graph(source)['receipt'])
                        (run / 'summary.json').write_text(json.dumps(summary))
                    if seed == 2 and capped_second:
                        continue
                    final_counts = copy.deepcopy(initial['runtime']['counts'])
                    for _ in range(2):
                        ex.increment(final_counts, 0, 1024, False)
                        ex.increment(final_counts, 1, 1024, True)
                    for arm, weight in replication.WEIGHTS.items():
                        run = seedroot / 'continuations/node_neighbors/lp' / arm
                        run.mkdir(parents=True)
                        final = copy.deepcopy(initial)
                        final['runtime']['counts'] = copy.deepcopy(final_counts)
                        final['runtime']['logical_step'] += 4
                        final['step'] += 4
                        ac.save(run / 'final.pt', final)
                        facebook = (.95 if weight == .1 else .89) if seed == 1 else .8
                        rows = [dict(additional_step=0, checkpoint=str(parent_run / 'endpoint.pt'),
                                     counts=initial['runtime']['counts'],
                                     validation=[dict(auc=.6, bce=.4), dict(auc=.8, bce=.2)]),
                                dict(additional_step=4, checkpoint=str(run / 'final.pt'), counts=final_counts,
                                     validation=[dict(auc=.85, bce=.2), dict(auc=facebook, bce=.2)])]
                        (run / 'history.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
                        summary = dict(status='complete', seed=seed, data_seed=0, probe_seed=0,
                                       start_sha256=ex.sha256(parent_run / 'endpoint.pt'),
                                       start_counts=initial['runtime']['counts'], teacher_sha256='shared_teacher',
                                       probe_sha256={s:ex.sha256(parent_run / f'probe_{s}.pt') for s in ac.SOURCES},
                                       surviving_counts=final_counts, budget_steps=4,
                                       protocol=dict(kd_weight=weight, learning_rate=.0005, weight_decay=1e-5,
                                                     schedule='alternating_1_to_1', validation_interval=2,
                                                     kd_temperature=1., optimizer='restored AdamW; no reset'))
                        (run / 'summary.json').write_text(json.dumps(summary))
                args = SimpleNamespace(root=str(root), additional_steps=4, selection_grid=2,
                                       development_parent=str(development))
                with contextlib.redirect_stdout(io.StringIO()), \
                     patch.object(ac.base, 'evaluate', side_effect=AssertionError('selection must not evaluate downstream')):
                    replication.prepare_eval(args)
                first = json.loads((root / 'seed1/evaluation/evaluation_manifest.json').read_text())
                second = json.loads((root / 'seed2/evaluation/evaluation_manifest.json').read_text())
                self.assertEqual(first['singleton_source_auc'][ac.SOURCES[1]], .9)
                self.assertEqual(second['singleton_source_auc'][ac.SOURCES[1]], .7)
                first_ids = {row['run_id'] for row in first['models']}
                self.assertIn('kd_w010_selected', first_ids)
                self.assertNotIn('kd_w100_selected', first_ids)
                self.assertEqual(first['unavailable'][0]['arm'], 'kd_w100')
                if capped_second:
                    self.assertEqual(len(second['models']), 2)
                    self.assertEqual(len(second['unavailable']), 2)
                    self.assertFalse(second['audit']['paired_continuation_samplers_match'])
                    self.assertEqual(second['audit']['parent_stop_reason'], 'safety_cap')
                else:
                    selected = next(row for row in second['models'] if row['run_id'] == 'kd_w010_selected')
                    self.assertEqual(selected['additional_step'], 4)
                    self.assertEqual(selected['validation'][1]['auc'], .8)
                    self.assertTrue(second['audit']['paired_continuation_samplers_match'])


if __name__ == '__main__':
    unittest.main()
