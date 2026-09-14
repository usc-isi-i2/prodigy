import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mixture_scaling import async_seed_singletons as m


def graph():
    return dict(x=torch.randn(7, 1536, generator=torch.Generator().manual_seed(42)),
                positive=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                validation=torch.tensor([[0, 1], [1, 2]]),
                sampler=m.ac.base.lp.ExactNonedges(7, torch.tensor([1, 9, 17])),
                receipt={'seed': 0})


class SeedSingletonTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_original_selection_and_patience_are_distinct(self):
        state = dict(best_bce=float('inf'), reference_bce=float('inf'), stale=0)
        self.assertTrue(m.update_selection(state, {'bce': .5, 'auc': .7}, 2000))
        self.assertTrue(m.update_selection(state, {'bce': .49999, 'auc': .71}, 4000))
        self.assertEqual(state['selected_step'], 4000)
        self.assertEqual(state['reference_bce'], .5)
        self.assertEqual(state['stale'], 1)
        self.assertFalse(m.update_selection(state, {'bce': .49999, 'auc': .8}, 6000))
        self.assertEqual(state['selected_step'], 4000)
        self.assertEqual(state['stale'], 2)
        self.assertTrue(m.update_selection(state, {'bce': .499, 'auc': .7}, 8000))
        self.assertEqual(state['stale'], 0)
        # The original implementation does not count stalled checks before 2500.
        m.update_selection(state, {'bce': .6, 'auc': .5}, 2400)
        self.assertEqual(state['stale'], 0)

    def test_seed_changes_training_but_keeps_data_probe_and_selection_policy(self):
        @contextlib.contextmanager
        def tracking(*_):
            yield SimpleNamespace(summary={}), lambda *_: None

        artifacts = []
        with tempfile.TemporaryDirectory() as tmp:
            for seed in (1, 2):
                args = SimpleNamespace(root=str(Path(tmp) / str(seed)), seed=seed, data_seed=0,
                                       config='unused', max_steps=4, validation_interval=2,
                                       patience=3, log_interval=2)
                reports = [dict(validation={'bce': bce, 'auc': .8},
                                training_probe={'bce': .3, 'auc': .9}) for bce in (.01, .5, .4)]
                with patch.object(m, 'load_config', return_value={}), \
                        patch.object(m.ac.base, 'load_graph', side_effect=lambda *_: graph()) as loading, \
                        patch.object(m.ac, 'tracked_run', side_effect=tracking), \
                        patch.object(m, 'measure', side_effect=reports), \
                        patch.object(m.ac, 'fixed_probe', wraps=m.ac.fixed_probe) as probing:
                    m.train(m.ac.SOURCES[1], args, torch.device('cpu'))
                self.assertEqual(loading.call_args.args[2], 0)
                self.assertEqual(probing.call_args.args[1], 813719 + 1009)
                root = Path(args.root) / 'singletons' / m.ac.SOURCES[1]
                summary = json.loads((root / 'summary.json').read_text())
                self.assertEqual(summary['selected_step'], 4)
                self.assertEqual(summary['selected_validation']['bce'], .4)
                self.assertEqual(summary['final_step'], 4)
                self.assertFalse(summary['converged'])
                self.assertEqual(summary['counts']['supervised_updates'], 4)
                best = torch.load(root / 'best.pt', weights_only=False)
                self.assertEqual(best['step'], 4)
                self.assertTrue(best['exact_sampling_resume'])
                self.assertEqual(best['runtime']['selection']['selected_step'], 4)
                artifacts.append((torch.load(root / 'checkpoints/step_0.pt', weights_only=False),
                                  torch.load(root / 'training_probe.pt', weights_only=False)))
            first, second = artifacts
            self.assertTrue(any(not torch.equal(first[0]['model'][k], second[0]['model'][k])
                                for k in first[0]['model']))
            for a, b in zip(first[1], second[1]):
                self.assertTrue(torch.equal(a[0], b[0]))
                self.assertTrue(torch.equal(a[1], b[1]))
            self.assertFalse(torch.equal(first[0]['samplers'][0]['generator'],
                                         second[0]['samplers'][0]['generator']))

    def test_data_seed_change_rejected(self):
        with self.assertRaisesRegex(ValueError, 'seed zero'):
            m.train(m.ac.SOURCES[0], SimpleNamespace(data_seed=1), torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
